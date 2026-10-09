"""v2 copy of ``tests/numsim/runtime/test_sync_instruction_predicates.py::test_pending_count_instruction_predicates``.

W6 triage: the invalid (non-``.noComplete``) pending-count case is still an
``error``; the finding is ``sync_protocol_error`` "pending_count: NotNoComplete"
(sync-behaviour-deltas M14), so the copy matches the structured ``context``
field naming ``NotNoComplete`` instead of the legacy ``noComplete`` text. The undefined lanes of the non-preserving
predicated call are not compared (see the test). Kernel copied verbatim.
"""

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def pending_predicate_kernel(*, drop=False, valid=True):
    action = "arrive_drop" if drop else "arrive"
    return tvm.script.from_source(
        f"""@T.prim_func
def pending(output: T.Buffer((3, 32), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_shared((1,), "uint64", align=8)
    state = T.alloc_local((1,), "uint64")
    result = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 2)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["mbarrier.{action}{".noComplete" if valid else ""}.shared.b64"](
            state[0], barrier.ptr_to([0]), T.uint32(1))
    result[0] = T.uint32(123)
    T.ptx.mbarrier.pending_count.b64(result[0], state[0], pred=lane == 0, preserve_dst=True)
    output[0, lane] = result[0]
    result[0] = T.uint32(123)
    T.ptx.mbarrier.pending_count.b64(result[0], state[0], pred=lane == 0)
    output[1, lane] = result[0]
    result[0] = T.uint32(123)
    T.ptx.mbarrier.pending_count.b64(result[0], state[0], pred=False, preserve_dst=True)
    output[2, lane] = result[0]
""",
        {"T": T},
    )


def pending_predicate_expected():
    expected = np.full((3, 32), 123, np.uint32)
    expected[1] = 0  # Only lane 0 is defined on GPU for the non-preserving call.
    expected[:2, 0] = 2
    return expected


def test_pending_count_instruction_predicates():
    for drop in (False, True):
        kernel = pending_predicate_kernel(drop=drop)
        inputs = {"output": np.zeros((3, 32), np.uint32)}
        for checker in (v2.synccheck, v2.racecheck):
            checker(kernel, dict(inputs)).require_clean()
            invalid = checker(pending_predicate_kernel(drop=drop, valid=False), dict(inputs))
            assert invalid.verdict == "error", invalid.format()
            if checker is v2.synccheck:
                assert any(
                    "NotNoComplete" in f.details.get("context", "") for f in invalid.findings
                ), invalid.format()
        actual = v2.Engine().run(v2.transpile(kernel), dict(inputs)).outputs["output"]
        expected = pending_predicate_expected()
        # Row 1 is the non-preserving predicated call: only lane 0 is defined on
        # the GPU (legacy comment). Legacy zeroed the other lanes; v2 keeps the
        # destination (guarded-Ptx keep_dst, CONTRACT_REQUESTS W2-20 item 1), so
        # only the defined lane is compared there.
        np.testing.assert_array_equal(actual[[0, 2]], expected[[0, 2]])
        assert actual[1, 0] == expected[1, 0]
