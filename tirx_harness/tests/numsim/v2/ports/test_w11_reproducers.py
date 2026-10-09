"""Smallest reproducers for the v2 bugs found by the W11 public-API
``other-assertion`` triage (CONTRACT_REQUESTS.md "W11-other-assertion").

The four W9-public-API bugs re-checked by this triage (``discard`` alignment,
``st.bulk`` size, ``isspacep.shared::cta`` rank, generic shared window bits) were
fixed by W2 while it ran; their legacy tests pass unchanged and need no copy here.

Each test asserts the correct (PTX ISA / device-validated / legacy-agreeing)
behaviour. All three are fixed and pass: W11-1 (``.pred`` bridge) and W11-3
(committed dynamic shared size) in lowering, W11-2 (per-signature PTX op
resolution) in the interpreter (W2 Rust scenario ``ptx_op_per_signature``).
W11-4 (128-bit destination carrier of a narrow ``cvt``), uncovered by the W11-2
fix, is fixed in oplib (W4 unit test ``integer_cvt_extends_into_every_carrier_width``).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


# -- W11-1 [W1 lowering]: a guarded op with a .pred destination ---------------


@T.prim_func
def guarded_pred_destination(output: T.Buffer((4, 32), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    d = T.alloc_local((4,), "uint32")
    for i in T.unroll(4):
        d[i] = T.uint32(91)
    # Every lane is predicated off.
    T.ptx["setp.lt.s32"](d[0], T.int32(1), T.int32(2), pred=lane < 0, preserve_dst=True)
    T.ptx["setp.lt.s32"](d[1], T.int32(1), T.int32(2), pred=lane < 0)
    T.ptx["testp.normal.f32"](d[2], T.float32(1), pred=lane < 0, preserve_dst=True)
    T.ptx["set.lt.u32.s32"](d[3], T.int32(1), T.int32(2), pred=lane < 0, preserve_dst=True)
    for i in T.unroll(4):
        output[i, lane] = d[i]


def test_w11_1_guarded_pred_destination_is_a_boolean_carrier():
    result = v2.Engine().run(v2.transpile(guarded_pred_destination), {"output": np.zeros((4, 32), np.uint32)})
    out = result.outputs["output"]
    # TVM's `_pred_keep` helper: `setp.ne.b32 pd, %0, 0; @p setp... pd; selp.b32 %0, 1, 0, pd`.
    np.testing.assert_array_equal(out[0], 1)
    np.testing.assert_array_equal(out[2], 1)
    # `_pred_undef`: `selp.b32 %0, 1, 0, pd` of an unwritten predicate is 0 or 1, never 91.
    assert set(out[1].tolist()) <= {0, 1}
    # A plain uint32 destination keeps its value (numsim-behaviour-deltas P8).
    np.testing.assert_array_equal(out[3], 91)


# -- W11-2 [W2 interp]: one Ptx op key resolved with its first use's types ----


@T.prim_func
def signed_cvt_into_several_carriers(output: T.Buffer((3,), "int64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    src = T.alloc_local((1,), "int8")
    src[0] = T.int8(-1)
    h = T.alloc_local((1,), "int16")
    w = T.alloc_local((1,), "int32")
    q = T.alloc_local((1,), "int64")
    T.ptx["cvt.s8.s8"](h[0], src[0])
    T.ptx["cvt.s8.s8"](w[0], src[0])
    T.ptx["cvt.s8.s8"](q[0], src[0])
    if lane == 0:
        output[0] = T.Cast("int64", h[0])
        output[1] = T.Cast("int64", w[0])
        output[2] = q[0]


def test_w11_2_ptx_op_resolves_per_use_carrier_types():
    result = v2.Engine().run(
        v2.transpile(signed_cvt_into_several_carriers), {"output": np.zeros(3, np.int64)}
    )
    # cvt.s8 sign-extends into every signed carrier (legacy: -1, -1, -1).
    np.testing.assert_array_equal(result.outputs["output"], [-1, -1, -1])


# -- W11-3 [W1 lowering]: dynamic pool size ignores the committed size ---------


@T.prim_func
def explicit_shared_strides_oob():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.SMEMPool()
    scratch = pool.alloc((2, 4, 64), "float32", strides=(8192, 64, 1), align=16)
    pool.commit()
    if lane == 0:
        scratch[1, 0, 0] = T.float32(1)


def test_w11_3_pool_access_beyond_committed_dyn_smem_is_out_of_bounds():
    # The pool commits prod(shape) * 4 = 2048 bytes; scratch[1, 0, 0] is byte 32768.
    with pytest.raises(v2.ExecutionError) as info:
        v2.Engine().run(v2.transpile(explicit_shared_strides_oob), {})
    assert "out_of_bounds" in str(info.value) or "out-of-bounds" in str(info.value)


# -- W11-4 [W4 oplib]: a narrow cvt into a 128-bit carrier --------------------


@T.prim_func
def narrow_cvt_into_b128_carrier(output: T.Buffer((2,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    src = T.alloc_local((1,), "int8")
    src[0] = T.int8(-1)
    wide = T.alloc_local((1,), "int128")
    T.ptx["cvt.s8.s8"](wide[0], src[0])
    if lane == 0:
        bytes_ = wide.view("uint64")
        output[0] = bytes_[0]
        output[1] = bytes_[1]


def test_w11_4_narrow_cvt_extends_into_a_128_bit_carrier():
    result = v2.Engine().run(v2.transpile(narrow_cvt_into_b128_carrier), {"output": np.zeros(2, np.uint64)})
    # Sign extension to the full register width, as legacy computes for int128/uint128 carriers.
    np.testing.assert_array_equal(result.outputs["output"], [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF])
