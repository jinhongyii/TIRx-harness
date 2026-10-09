"""v2 delta copy of ``tests/numsim/runtime/test_mbarrier_maintenance.py::test_maintenance_active_addresses_still_checked``.

Deltas:

- Finding kind ``oob`` is now ``out_of_bounds``:
  ``docs/development/racecheck-behaviour-deltas.md`` P5 (racecheck) and T6, and the
  synccheck rename listed in ``docs/development/test-migration.md`` ("Kind names").
  The message names the byte span (``out-of-bounds access [8, 16) to ... (smem...)``)
  instead of legacy's ``raw shared address 8``; same address, same lane.
- An additional ``review`` ``uninitialized_read`` on a register: the address
  operand ``address[0]`` is a tracked local that only the selected lane wrote, and
  TVM binds it as an inline-asm input for every lane (the ``pred=`` guard is the
  ``@p`` inside the asm), so the other lanes read an indeterminate register. v2
  reports register-space uninitialized reads (CONTRACT_REQUESTS.md V2C-19/V2C-20,
  "Tracked locals are ``Space::Reg``"; W2-20 table row "register-space uninit
  reports"); legacy did not. It is review-only and does not change the verdict.

Kept: the predicated-on lane's out-of-bounds maintenance address is still an
error in both checkers, for ``check_layout`` (layout 0) and ``inval`` (layout None).
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def maintenance_case(preserve=True, layout=1, generic=False, *, invalid=False, count=1):
    """Copied verbatim from the legacy test module."""
    pointer = f"barrier.ptr_to([{int(invalid)}])"
    address = (
        f'T.reinterpret("uint64", {pointer})'
        if generic
        else f"T.cuda.cvta_generic_to_shared({pointer})"
    )
    space = "" if generic else ".shared::cta"
    operation = (
        f'T.ptx["mbarrier.check_layout.layout::v{layout}{space}.b64"]('
        f"value[0], address[0], pred=lane == selected, preserve_dst={preserve})"
        if layout is not None
        else f"""T.ptx["mbarrier.inval{space}.b64"](address[0], pred=lane == selected)
    if lane == selected:
        T.ptx["mbarrier.init.layout::v1.shared.b64"](barrier.ptr_to([0]), {count})
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]), T.Cast("uint32", T.if_then_else(selected < 0, 2, {count})))
    T.cuda.cta_sync()
    T.ptx.mbarrier.test_wait.parity.shared.b64(value[0], barrier.ptr_to([0]), T.uint32(0))
    if {count} != 1:
        layout_ok = T.alloc_local((1,), "uint32")
        T.ptx["mbarrier.check_layout.layout::v1.shared::cta.b64"](layout_ok[0], barrier.ptr_to([0]))
        value[0] = value[0] * T.Cast("uint32", layout_ok[0] == T.Cast("uint32", selected >= 0))"""
    )
    output = (
        'T.Cast("uint32", value[0])'
        if preserve or layout is None
        else 'T.if_then_else(lane == selected, T.Cast("uint32", value[0]), T.uint32(0))'
    )
    return tvm.script.from_source(
        f'''@T.prim_func
def kernel(output: T.Buffer((32,), "uint32"), selected: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_shared((1,), "uint64", align=8)
    address = T.alloc_local((1,), "{"uint64" if generic else "uint32"}")
    value = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 2)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == selected:
        address[0] = {address}
    value[0] = T.Cast("uint32", T.if_then_else(lane % 2 == 0, -7, 0))
    {operation}
    output[lane] = {output}
''',
        {"T": T},
    )


def test_maintenance_active_addresses_still_checked():
    """Delta copy; see the module docstring (P5/T6 kind rename, register uninit review)."""

    for layout in (0, None):
        kernel = maintenance_case(layout=layout, invalid=True)
        for checker in (v2.synccheck, v2.racecheck):
            report = checker(kernel, {"output": np.zeros(32, np.uint32), "selected": 0})
            assert report.verdict == "error", report.format()
            errors = [f for f in report.findings if f.status == "error"]
            assert [f.kind for f in errors] == ["out_of_bounds"], report.format()
            assert "out-of-bounds access [8, 16)" in errors[0].message
            assert "smem" in errors[0].message
            others = [f for f in report.findings if f.status != "error"]
            assert all(
                f.status == "review"
                and f.kind == "uninitialized_read"
                and f.details["space"] == "reg"
                for f in others
            ), report.format()
