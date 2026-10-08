"""v2 expected-error copy of
``tests/numsim/runtime/test_tf32_layout_f_poison.py::test_tf32_layout_f_unwritten_holes_are_zero_filled_and_require_review``.

Ruling: ``numsim-behaviour-deltas.md`` row **T4** (CONTRACT_REQUESTS.md "Contract
changes for W1" Batch 2 item 17; W2 engine-stops triage "TMEM buffer access outside
the warp's sub-partition"). Warp 0 stages the layout-F A operand with
``tmem[physical_lane, k]``, ``physical_lane = (row // 16) * 32 + row % 16`` up to
111, so warp 0 lane 16 stores TMEM lane 32. A buffer-form TMEM store executes as
``tcgen05.st .32x32b`` and may address only the warp's own sub-partition: v2 stops
with ``bad_address`` before the MMA. Legacy modelled TMEM abstractly and pinned the
zero-filled holes (``verdict == "review"``, ``uninitialized_read``), which this
kernel can no longer reach. Kernel and operand builder copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def raw_tf32_layout_f_hole_poison(
    consume_holes: T.int32,
    a: T.Buffer((64, 8), "float32"),
    b_physical: T.Buffer((4096,), "uint8"),
    valid: T.Buffer((64, 32), "float32"),
    holes: T.Buffer((4, 16), "uint32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    readback = T.alloc_local((1,), "uint32")
    desc_i: T.uint32
    desc_b: T.uint64

    thread = warp * 32 + lane
    for copy_i in T.serial(32):
        offset = thread + copy_i * 128
        shared_b[offset] = b_physical[offset]

    if warp == 0:
        for row_group in T.unroll(2):
            row = row_group * 32 + lane
            physical_lane = (row // 16) * 32 + row % 16
            for k in T.unroll(8):
                tmem[physical_lane, k] = T.reinterpret("uint32", a[row, k])
            for col in T.serial(32):
                tmem[physical_lane, 16 + col] = T.reinterpret("uint32", T.float32(4))
    if lane < 8:
        tmem[warp * 32 + 16 + lane, 16] = T.uint32(0x3F400000)
    T.cuda.cta_sync()

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
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(16),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
            pred=T.cast(lane == 7, "uint32"),
        )
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
        readback[0], T.cuda.get_tmem_addr(T.uint32(0), 0, 16)
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()

    if warp < 2:
        row = warp * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.serial(32):
            valid[row, col] = T.reinterpret("float32", tmem[physical_lane, 16 + col])
    if consume_holes != 0:
        if lane >= 16:
            holes[warp, lane - 16] = readback[0]


def _kmajor_swizzle_physical(logical_bytes: np.ndarray) -> np.ndarray:
    rows, row_bytes = logical_bytes.shape
    swizzle_len = 3
    stride_bytes = 1024
    physical = np.zeros(((rows + 7) // 8) * stride_bytes, dtype=np.uint8)
    row_stride = 16 << swizzle_len
    swizzle_mask = (1 << swizzle_len) - 1
    for row in range(rows):
        for byte_in_row in range(row_bytes):
            atom = byte_in_row // 16
            byte_in_atom = byte_in_row % 16
            unswizzled = (
                (row % 8) * row_stride + (row // 8) * stride_bytes + atom * 16 + byte_in_atom
            )
            atom_index = unswizzled >> 4
            swizzled_atom = atom_index ^ (
                (atom_index & (swizzle_mask << swizzle_len)) >> swizzle_len
            )
            physical[(swizzled_atom << 4) | byte_in_atom] = logical_bytes[row, byte_in_row]
    return physical


def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _assert_tmem_subpartition_bad_address(excinfo, buffer: str, warp: int) -> None:
    """Row T4: the stopping diagnostic is an ``error`` of kind ``bad_address`` for a
    TMEM lane outside the issuing warp's 32-lane sub-partition."""

    stop = _first_stop(excinfo.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "bad_address", stop
    message = str(stop.get("message", ""))
    assert f"{buffer}[" in message, stop
    assert f"outside warp {warp}'s sub-partition" in message, stop

def test_tf32_layout_f_unwritten_holes_are_zero_filled_and_require_review():
    """Replaces ``tests/numsim/runtime/test_tf32_layout_f_poison.py::test_tf32_layout_f_unwritten_holes_are_zero_filled_and_require_review``.
    Delta row T4: ``bad_address`` on warp 0's store to ``tmem`` lane 32."""

    row = np.arange(64, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    col = np.arange(32, dtype=np.float32)[:, None]
    a = ((row % np.float32(5)) - np.float32(2)) * np.float32(0.5) + k * np.float32(0.25)
    b = ((col % np.float32(7)) - np.float32(3)) * np.float32(0.25) - k * np.float32(0.5)
    b_bytes = np.ascontiguousarray(b).view(np.uint8).reshape(32, 32)

    module = v2.transpile(raw_tf32_layout_f_hole_poison)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {
                "consume_holes": np.int32(1),
                "a": np.ascontiguousarray(a),
                "b_physical": _kmajor_swizzle_physical(b_bytes),
                "valid": np.zeros((64, 32), dtype=np.float32),
                "holes": np.zeros((4, 16), dtype=np.uint32),
            },
        )
    _assert_tmem_subpartition_bad_address(excinfo, "tmem", 0)
    assert "tmem lane 32" in str(_first_stop(excinfo.value).get("message", ""))
