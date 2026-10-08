"""Smallest reproducers for the v2 bugs found by the W11 public-API
``other-assertion`` triage (CONTRACT_REQUESTS.md "W11-other-assertion").

The four W9-public-API bugs re-checked by this triage (``discard`` alignment,
``st.bulk`` size, ``isspacep.shared::cta`` rank, generic shared window bits) were
fixed by W2 while it ran; their legacy tests pass unchanged and need no copy here.

Each test asserts the correct (PTX ISA / device-validated / legacy-agreeing)
behaviour and is marked ``v2_gap`` until its owner fixes v2; an XPASS means the
gap is closed and the marker (and the legacy test's blocked status) can go.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
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


@v2_gap(
    "W11-1: a guarded op's .pred destination carried in a uint32 is kept raw (91), not as (old != 0); "
    "same root cause as CONTRACT_REQUESTS W9-public-API phase 6 [W1] .pred bridge (ptx_decode drops p<i>)"
)
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


@v2_gap("W11-2: Loaded::new resolves each Ptx op with the operand types of its first use only")
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


@v2_gap("W11-3: a shared.dyn pool is sized from its views' strided extents, not tirx.dyn_smem_bytes")
def test_w11_3_pool_access_beyond_committed_dyn_smem_is_out_of_bounds():
    # The pool commits prod(shape) * 4 = 2048 bytes; scratch[1, 0, 0] is byte 32768.
    with pytest.raises(v2.ExecutionError) as info:
        v2.Engine().run(v2.transpile(explicit_shared_strides_oob), {})
    assert "out_of_bounds" in str(info.value) or "out-of-bounds" in str(info.value)
