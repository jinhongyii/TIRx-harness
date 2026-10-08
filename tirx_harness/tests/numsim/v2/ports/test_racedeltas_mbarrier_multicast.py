"""v2 copies of ``tests/numsim/runtime/test_mbarrier_multicast.py::
test_mbarrier_multicast32`` and ``::test_mbarrier_multicast_lane_masks`` with
racecheck delta **B7**.

CTA 0 issues a qualifier-less ``mbarrier.<form>.shared::cluster.
multicast::cluster::32b`` whose mask names CTA 0 and the peer CTA 19 (or, in
the lane-mask test, lane 0 -> CTA 0 and lane 1 -> CTA 19). Legacy required
both checkers clean. Under B7 every *arrive* form defaults to
``.release.cta``; the copy delivered to CTA 19's barrier does not include
CTA 19's waiter, so Racecheck reports exactly two ``scope_mismatch`` errors,
release warp 0 (the multicast arrive) against CTA 19's (warp 19) two
``.acquire.cta`` observations of that phase: ``mbarrier_wait(.., 0)`` and the
following ``mbarrier.test_wait``. The delivery to CTA 0 itself is same-CTA
and fine. No follow-on race exists (the kernel exchanges no data through the
barrier; ``cluster_sync`` orders everything else). The non-arrive forms
(``expect_tx``, ``complete_tx``) carry no release and stay clean. Synccheck
is clean and every output is unchanged. The kernel builder is copied verbatim
from the legacy file (returning the source text as well).
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

from ._racedeltas import assert_b7_scope_mismatch

pytestmark = requires_v2_engine

MULTICAST_FORMS = (
    "arrive",
    "arrive_nocount",
    "arrive_drop",
    "arrive_drop_nocount",
    "arrive_expect_tx",
    "arrive_drop_expect_tx",
    "expect_tx",
    "complete_tx",
)


def multicast_barrier_source(form, *, mask=None, ctas=20, lane_varying=False):
    arrival = form.startswith("arrive")
    drop = "drop" in form
    transactions = "tx" in form
    initial = 2 if arrival else 1
    action = form.replace("_nocount", "").replace("_expect_tx", ".expect_tx")
    instruction = f"mbarrier.{action}.shared::cluster.multicast::cluster::32b.b64"
    args = ["barrier.ptr_to([0])"]
    if not form.endswith("_nocount"):
        args.append(f"T.uint32({16 if transactions else 1})")
    args.append(
        f"T.uint32(1) << (lane * {ctas - 1})"
        if lane_varying
        else f"T.uint32({(1 << (ctas - 1)) | 1 if mask is None else mask})"
    )
    return f"""
@T.prim_func
def multicast_barrier(out: T.Buffer(({ctas}, 2), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([{ctas}])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), {initial})
        {"T.ptx.mbarrier.expect_tx.shared.b64(barrier.ptr_to([0]), 16)" if form == "complete_tx" else "T.evaluate(0)"}
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if cta == 0 and lane < {2 if lane_varying else 1}:
        T.ptx["{instruction}"]({", ".join(args)})
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0 or cta == {ctas - 1}:
            {"T.ptx.mbarrier.complete_tx.relaxed.cta.shared.b64(barrier.ptr_to([0]), 16)" if transactions and form != "complete_tx" else "T.evaluate(0)"}
            T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]), T.uint32(1))
        else:
            {"T.ptx.mbarrier.complete_tx.relaxed.cta.shared.b64(barrier.ptr_to([0]), 16)" if form == "complete_tx" else "T.evaluate(0)"}
            T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]), T.uint32({initial}))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
        out[cta, 0] = ready[0]
        T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]),
            T.uint32({1 if drop else initial}) if cta == 0 or cta == {ctas - 1} else T.uint32({initial}))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 1)
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
        out[cta, 1] = ready[0]
    T.cuda.cluster_sync()
"""


def _inputs():
    return {"out": np.zeros((20, 2), np.uint32)}


@pytest.mark.parametrize("form", MULTICAST_FORMS)
def test_mbarrier_multicast32(form, tmp_path):
    """Replaces ``tests/numsim/runtime/test_mbarrier_multicast.py::test_mbarrier_multicast32`` (all 8 forms).

    Legacy ``run_checked``: both checkers clean, then ``out == 1``. Arrive
    forms: racecheck delta B7, exactly two ``scope_mismatch`` (warp 0's
    qualifier-less multicast arrive vs. CTA 19's ``mbarrier_wait`` and
    ``mbarrier.test_wait``). ``expect_tx`` / ``complete_tx``: clean as legacy.
    """

    source = multicast_barrier_source(form)
    kernel = tvm.script.from_source(source, {"T": T})
    v2.synccheck(kernel, _inputs()).require_clean()
    report = v2.racecheck(kernel, _inputs())
    if form.startswith("arrive"):
        assert_b7_scope_mismatch(report, acquire_warps={19}, count=2, source=source)
    else:
        report.require_clean()
    result = v2.Engine().run(v2.transpile(kernel, cache_dir=tmp_path), _inputs())
    np.testing.assert_array_equal(result.outputs["out"], np.ones((20, 2), np.uint32))


def test_mbarrier_multicast_lane_masks(tmp_path):
    """Replaces ``tests/numsim/runtime/test_mbarrier_multicast.py::test_mbarrier_multicast_lane_masks``.

    Lane 0 multicasts ``arrive_drop`` to CTA 0, lane 1 to CTA 19. Racecheck
    delta B7: exactly two ``scope_mismatch`` against CTA 19's waits (CTA 0's
    delivery is same-CTA). Synccheck clean and ``out == 1`` unchanged.
    """

    source = multicast_barrier_source("arrive_drop", lane_varying=True)
    kernel = tvm.script.from_source(source, {"T": T})
    v2.synccheck(kernel, _inputs()).require_clean()
    assert_b7_scope_mismatch(v2.racecheck(kernel, _inputs()), acquire_warps={19}, count=2, source=source)
    result = v2.Engine().run(v2.transpile(kernel, cache_dir=tmp_path), _inputs())
    np.testing.assert_array_equal(result.outputs["out"], 1)
