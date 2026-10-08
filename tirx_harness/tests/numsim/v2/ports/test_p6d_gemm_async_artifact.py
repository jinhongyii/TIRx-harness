"""v2 expected-error ports of the TMEM sub-partition items of
``tests/numsim/integration/test_gemm_async_artifact.py``.

Ruling: ``docs/development/numsim-behaviour-deltas.md`` row T4 (and
``CONTRACT_REQUESTS.md`` "Contract changes for W1", Batch 2 item 17; W2
engine-stops triage, "TMEM buffer access outside the warp's sub-partition").
A buffer-form TMEM ``Load``/``Store`` executes as ``tcgen05.ld/st .32x32b``,
so a warp may only address TMEM lanes ``32 * (warp_id % 4) .. +32``. Every
kernel below runs one warp (warp 0) and reads or writes TMEM lanes 32..127
through a TMEM buffer, so v2 stops with a ``bad_address`` execution error
(``<buf>[i]: tmem lane L is outside warp 0's sub-partition``). Legacy
modelled TMEM buffers abstractly and returned the numerical result.

Kernels are copied verbatim from the legacy module;
``dense_gemm_async_tmem_a_transposed_b`` comes from the pure-TVM
``tests/numsim/support/kernels.py``. The numerical assertions of the legacy
tests are unreachable (the run stops), so each copy asserts the exception
type, the stopping diagnostic (status ``error``, kind ``bad_address``) and the
sub-partition message.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, TCol, TileLayout, TLane, tmem_datapath_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout

from tests.numsim.support.kernels import dense_gemm_async_tmem_a_transposed_b
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_MMA_BF16_64X64 = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, (64, 64))
_TMEM_WRONG_F_GROUP_ORDER = TileLayout(
    S[(2, 2, 16, 8) : (32 @ TLane, 64 @ TLane, 1 @ TLane, 1 @ TCol)]
)
_TMEM_CTA2_BANKED_A = TileLayout(S[(2, 64, 16) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])


def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _assert_subpartition_stop(kernel, inputs, *, buffer: str, **engine_kwargs) -> None:
    """Row T4: the run stops with ``bad_address`` on a TMEM lane outside warp 0's sub-partition."""

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine(**engine_kwargs).run(v2.transpile(kernel), inputs)
    stop = _first_stop(excinfo.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "bad_address", stop
    assert f"{buffer}[" in str(excinfo.value), excinfo.value
    assert "outside warp 0's sub-partition" in str(excinfo.value), excinfo.value



@T.prim_func
def dense_gemm_async_m64_layout_f(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(64):
            for col in T.serial(8):
                accumulator[row, col] = T.float32(-777)
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=True)
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_bf16_m64_layout_f(
    left: T.Buffer((64, 16), "bfloat16"),
    right: T.Buffer((8, 16), "bfloat16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "bfloat16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "bfloat16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_m64_packed_layout_e_inferred(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("E", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_bf16_large_library_path(
    left: T.Buffer((64, 64), "bfloat16"),
    right: T.Buffer((64, 64), "bfloat16"),
    output: T.Buffer((64, 64), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", layout=_MMA_BF16_64X64)
    right_shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", layout=_MMA_BF16_64X64)
    accumulator = T.decl_buffer(
        (64, 64),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 64),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        for row in T.serial(64):
            for col in T.serial(64):
                output[row, col] = accumulator[row, col]



@T.prim_func
def dense_gemm_async_accumulation_rounding(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(64):
            for col in T.serial(8):
                accumulator[row, col] = T.float32(16777216)
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=True)
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_m64_wrong_f_group_order(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=_TMEM_WRONG_F_GROUP_ORDER,
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(64):
            for col in T.serial(8):
                accumulator[row, col] = T.float32(-777)
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]




@T.prim_func
def dense_gemm_async_m64_cta2_banked_a(
    left: T.Buffer((2, 2, 64, 16), "float16"),
    right: T.Buffer((2, 16, 16), "float16"),
    output: T.Buffer((2, 64, 32), "float32"),
):
    """CTA2 Layout-B form whose two A banks select the two B shards."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_tmem = T.decl_buffer(
        (2, 64, 16),
        "float16",
        scope="tmem",
        layout=_TMEM_CTA2_BANKED_A,
        allocated_addr=0,
    )
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=32,
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




def test_dense_gemm_async_starts_the_fma_chain_from_input_d():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_starts_the_fma_chain_from_input_d``.

    Delta T4: the 64-row Layout-F accumulator initialisation by lane 0 reaches
    TMEM lane 32 -> ``bad_address``."""

    left = np.zeros((64, 16), dtype=np.float16)
    right = np.zeros((8, 16), dtype=np.float16)
    left[:, 0] = np.float16(-4096)
    left[:, 1] = np.float16(1)
    right[:, 0] = np.float16(4096)
    right[:, 1] = np.float16(0.5)
    _assert_subpartition_stop(
        dense_gemm_async_accumulation_rounding,
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
        buffer="accumulator",
    )


def test_m64_tcgen_mma_uses_layout_f_independently_of_declared_tmem_layout():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_m64_tcgen_mma_uses_layout_f_independently_of_declared_tmem_layout``.

    Delta T4: the Layout-F kernel initialises 64 TMEM rows from lane 0 of
    warp 0 -> ``bad_address``.

    The second legacy kernel (``dense_gemm_async_m64_wrong_f_group_order``,
    a TMEM accumulator layout that is not a tcgen05 datapath layout) no
    longer reaches the engine: v2 lowers tile ops through TVM's own schedule
    dispatch, which has no ``tirx.tile.gemm_async`` dispatch for that layout, so
    ``v2.transpile`` raises ``UnsupportedTIRxError``. Legacy transpiled it and
    pinned the permuted rows its abstract TMEM model produced; that
    permutation is not observable on hardware (TVM does not compile the
    kernel either), so the copy asserts the transpile-time rejection."""

    left = (np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 17 - 8).astype(np.float16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 11 - 5).astype(np.float16)
    _assert_subpartition_stop(
        dense_gemm_async_m64_layout_f,
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
        buffer="accumulator",
    )
    with pytest.raises(UnsupportedTIRxError, match="gemm_async"):
        v2.transpile(dense_gemm_async_m64_wrong_f_group_order)


def test_bf16_m64_tcgen_mma_uses_layout_f():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_bf16_m64_tcgen_mma_uses_layout_f``.

    Delta T4: the read-back of the 64-row Layout-F accumulator by lane 0
    reaches TMEM lane 32 -> ``bad_address``."""

    bf16 = ml_dtypes.bfloat16
    left = (np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 9 - 4).astype(bf16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 7 - 3).astype(bf16)
    _assert_subpartition_stop(
        dense_gemm_async_bf16_m64_layout_f,
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
        buffer="accumulator",
    )


def test_m64_tcgen_mma_infers_weight_stationary_from_packed_layout_e():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_m64_tcgen_mma_infers_weight_stationary_from_packed_layout_e``.

    Delta T4: the Layout-E accumulator read-back by lane 0 reaches TMEM lane
    64 -> ``bad_address``."""

    left = (np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 9 - 4).astype(np.float16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 7 - 3).astype(np.float16)
    _assert_subpartition_stop(
        dense_gemm_async_m64_packed_layout_e_inferred,
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
        buffer="accumulator",
    )


def test_large_bf16_gemm_async_uses_one_engine_gemm():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_large_bf16_gemm_async_uses_one_engine_gemm``.

    Delta T4: the 64x64 Layout-F accumulator read-back by lane 0 reaches TMEM
    lane 32 -> ``bad_address``."""

    bf16 = ml_dtypes.bfloat16
    left = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 5 - 2).astype(bf16)
    right = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 7 - 3).astype(bf16)
    _assert_subpartition_stop(
        dense_gemm_async_bf16_large_library_path,
        {"left": left, "right": right, "output": np.zeros((64, 64), dtype=np.float32)},
        buffer="accumulator",
    )


def test_dense_gemm_async_reads_tmem_a_and_transposed_b_storage():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_dense_gemm_async_reads_tmem_a_and_transposed_b_storage``.

    Delta T4: lane 0 fills the 128-row TMEM A operand and reaches TMEM lane 32
    -> ``bad_address``."""

    left = (np.arange(128 * 16, dtype=np.float32).reshape(128, 16) % 13 - 6).astype(np.float16)
    right_transposed = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 7 - 3).astype(
        np.float16
    )
    _assert_subpartition_stop(
        dense_gemm_async_tmem_a_transposed_b,
        {
            "left": left,
            "right_transposed": right_transposed,
            "output": np.zeros((128, 16), dtype=np.float32),
        },
        buffer="left_tmem",
    )


def test_cta_group2_banked_a_selects_matching_b_shard():
    """Replaces ``tests/numsim/integration/test_gemm_async_artifact.py::test_cta_group2_banked_a_selects_matching_b_shard``.

    Delta T4: lane 0 fills both 64-lane A banks and reaches TMEM lane 32 ->
    ``bad_address`` (``Engine(max_workers=2)`` kept)."""

    left = (np.arange(2 * 2 * 64 * 16, dtype=np.float32).reshape(2, 2, 64, 16) % 11 - 5).astype(
        np.float16
    )
    right = (np.arange(2 * 16 * 16, dtype=np.float32).reshape(2, 16, 16) % 7 - 3).astype(np.float16)
    _assert_subpartition_stop(
        dense_gemm_async_m64_cta2_banked_a,
        {"left": left, "right": right, "output": np.zeros((2, 64, 32), dtype=np.float32)},
        buffer="left_tmem",
        max_workers=2,
    )
