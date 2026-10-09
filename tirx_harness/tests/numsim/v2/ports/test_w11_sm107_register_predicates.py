"""v2 delta copy of ``tests/numsim/runtime/test_sm107_register_predicates.py::test_sm107_register_predicates``
(public-API ``other-assertion`` triage, W11).

Delta (numsim-behaviour-deltas P8): a predicated-off PTX op with a write-only
destination (``preserve_dst=False``) leaves the destination register as it was in
v2; legacy wrote zero. TVM renders that form with a write-only ``"=r"`` binding
(``tvm/backend/cuda/ptx/render.py`` ``_pred_undef`` helpers), so the C-level value
on an inactive lane is unspecified on hardware: the GPU microtest
``tests/numsim/microtests/test_sm107_register_predicates.py`` compares only the
active lanes when ``preserve_dst=False`` ("Write-only inactive GPU registers are
undefined, not necessarily zero"). Legacy's zero and v2's kept value are both
admissible representatives; v2 keeps the PTX-level ``@p`` semantics (an
unexecuted instruction does not write), CONTRACT_REQUESTS "W1 (2026-10-08): W2-20
lowering items" item 1.

Kept unchanged: every active-lane value (byte-wise ``set.lt.u8x4``, 2:4
``spcompress``/``spdecompress``), the ``preserve_dst=True`` inactive lanes (91),
and clean synccheck/racecheck for every mask. Changed: the ``preserve_dst=False``
inactive lanes keep the initial 91 instead of legacy's 0. All destinations are
plain ``uint32`` registers (no ``.pred`` carrier, whose bridge is a separate bug,
CONTRACT_REQUESTS "W11-other-assertion" item 1).
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def sm107_register_predicate_kernel(preserve):
    """Copied verbatim from the legacy test module."""
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(output: T.Buffer((5, 32), "uint32"), selected: T.uint32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    active = (selected & (T.uint32(1) << T.Cast("uint32", lane))) != T.uint32(0)
    src = T.alloc_local((5,), "uint32")
    dst = T.alloc_local((5,), "uint32")
    for i in T.unroll(5):
        dst[i] = 91
    if active:
        src[0] = T.uint32(0x03020401)
        src[1] = T.uint32(0x03020401)
        src[2] = T.uint32(0xDD)
        src[3] = T.uint32(0x03040304)
        src[4] = T.uint32(0)
    T.ptx["set.lt.u8x4"](dst[0], src[0], src[3], pred=active, preserve_dst={preserve})
    T.ptx["spcompress.b8.b2.sp::2:4.x1"](dst[1], dst[2], src[0], src[1], src[4],
        pred=active, preserve_dst={preserve})
    T.ptx["spdecompress.b8.b2.sp::2:4.x2"](dst[3], dst[4], src[2], src[3],
        pred=active, preserve_dst={preserve})
    for i in T.unroll(5):
        output[i, lane] = dst[i]
""",
        {"T": T},
    )


def sm107_register_predicate_expected(mask):
    """Legacy expectation, except that inactive lanes keep 91 for both
    ``preserve_dst`` values (delta P8; legacy wrote 0 when not preserving)."""
    expected = np.full((5, 32), 91, np.uint32)
    active = [lane for lane in range(32) if mask & (1 << lane)]
    # Byte-wise less-than; max-selection of [1,4,2,3] keeps indices 1,3;
    # decompression scatters [4,3,4,3] into those positions and zero-fills.
    expected[:, active] = np.array(
        [0x00FF00FF, 0xDD, 0x03040304, 0x03000400, 0x03000400], np.uint32
    )[:, None]
    return expected


@pytest.mark.parametrize("preserve", [False, True])
def test_sm107_register_predicates(preserve):
    """Delta copy; see the module docstring (P8: inactive write-only destinations keep their value)."""
    kernel = sm107_register_predicate_kernel(preserve)
    module = v2.transpile(kernel)
    for mask in (0, 0x80000000, 0xAAAAAAAA, 0xFFFFFFFF):
        inputs = {"output": np.zeros((5, 32), np.uint32), "selected": mask}
        for checker in (v2.synccheck, v2.racecheck):
            checker(kernel, dict(inputs)).require_clean()
        result = v2.Engine().run(module, dict(inputs))
        np.testing.assert_array_equal(result.outputs["output"], sm107_register_predicate_expected(mask))
