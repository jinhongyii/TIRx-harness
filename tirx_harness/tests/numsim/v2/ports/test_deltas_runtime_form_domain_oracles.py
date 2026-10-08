"""v2 copy of ``tests/numsim/runtime/test_runtime_form_domain_oracles.py::test_ordering_fence_and_register_policy_finite_domains_execute``.

W6 ruling: sync-behaviour-deltas setmaxnreg row R1 (checker-mode register pool
in every mode; NumSim reports ``InvalidDirection``). The kernel's
``setmaxnreg.dec`` to 32 while the warp holds 24 registers is an invalid
direction, so the run stops with ``sync_protocol_error``
``RegPool(InvalidDirection …)`` instead of producing the legacy output.
Kernel builder copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx  # noqa: F401
from tvm.tirx.layout import S, TCol, TileLayout, TLane, laneid, wg_local_layout, wid_in_wg  # noqa: F401

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def _make_complete_ordering_runtime_kernel():
    statements = [
        f"    T.ptx.fence.{semantics}.{scope}()"
        for semantics in ("sc", "acq_rel")
        for scope in ("cta", "cluster", "gpu", "sys")
    ]
    statements.extend(
        statement
        for increase in (False, True)
        for register_count in range(24, 257, 8)
        for statement in (
            "    T.ptx.setmaxnreg."
            f"{'inc' if increase else 'dec'}.sync.aligned.u32({register_count})",
            "    T.cuda.warpgroup_sync(7)",
        )
    )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def ordering_complete_runtime_domain(output: T.Buffer((1,), 'int32')):",
                "    T.device_entry()",
                "    warp = T.warp_id([4])",
                "    lane = T.lane_id([32])",
                *statements,
                "    if (warp == 0) and (lane == 0):",
                "        output[0] = 0x13579BDF",
            )
        ),
        extra_vars={"T": T},
    )

ORDERING_COMPLETE_RUNTIME_DOMAIN = _make_complete_ordering_runtime_kernel()


def test_ordering_fence_and_register_policy_finite_domains_execute():
    module = v2.transpile(ORDERING_COMPLETE_RUNTIME_DOMAIN)
    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine(max_workers=1).run(module, {"output": np.zeros(1, dtype=np.int32)})
    stops = [d for d in excinfo.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and stops[0]["status"] == "error", excinfo.value.diagnostics
    assert stops[0]["kind"] == "sync_protocol_error", stops[0]
    assert "invalid direction" in str(stops[0].get("message", "")).lower() or "InvalidDirection" in str(stops[0]), stops[0]
