"""v2 copy of ``tests/numsim/runtime/test_cp_async_mbarrier_visibility.py::
test_copy_arrival_is_not_an_explicit_commit`` with the W8-5 delta.

Legacy required a clean Synccheck verdict for every case. In the two racy
cases (``{}`` and ``commit`` with ``wait_group(1)``) the copy into
``shared[0:4]`` is in no completed (waited) group, so lane 0's
``output[0] = shared[0]`` is not ordered after the copy and, in the recorded
execution, runs before it lands. ``shared`` is a fresh shared allocation that
no thread ever stores to: the read genuinely observes uninitialized shared
bytes ``[0, 4)``. NumSim uninitialized reads use
``ValidityPolicy::ZeroAndReport`` in every mode (W8-5 in
``numsim-core/CONTRACT_REQUESTS.md``; ``docs/development/dev-loop.md``), so
both checkers now carry exactly that ``uninitialized_read`` ``review``
diagnostic: Synccheck's verdict is ``review`` and Racecheck keeps the legacy
``error`` race plus the same review. The clean cases (and their outputs) are
unchanged. The kernel is copied verbatim from the legacy file.
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim import v2
from tests.numsim.v2.checkers._runnable import requires_v2_engine

pytestmark = requires_v2_engine


def inputs():
    return {"source": np.arange(7, 11, dtype=np.uint32), "output": np.zeros(1, np.uint32)}


def arrived_open_batch_case(
    *, commit=False, pending=0, empty_tail=False, observe_barrier=False, overwrite_after_wait=False
):
    overwrite = (
        """T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        shared[0] = T.uint32(42)"""
        if overwrite_after_wait
        else "T.evaluate(0)"
    )
    source = f"""
@T.prim_func
def kernel(source: T.Buffer((4,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_shared((4,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=16)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx["cp.async.ca.shared.global"](shared.ptr_to([0]), source.ptr_to([0]), 16)
        T.ptx.cp.async_.mbarrier.arrive.noinc.shared.b64(barrier.ptr_to([0]))
        {overwrite}
        {"T.ptx.cp.async_.commit_group()" if commit else "T.evaluate(0)"}
        {"T.ptx.cp.async_.commit_group()" if empty_tail else "T.evaluate(0)"}
        T.ptx.cp.async_.wait_group({pending})
        {"T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)" if observe_barrier else "T.evaluate(0)"}
        output[0] = shared[0]
"""
    return tvm.script.from_source(source, {"T": T}), source


def _span_text(source: str, record: dict) -> str:
    span = record.get("source_span") or {}
    line = source.split("\n")[span["line"] - 1]
    return line[span["column"] - 1 : span["end_column"] - 1]


def _assert_only_shared0_uninitialized_reviews(report, source: str) -> None:
    """Every review is an ``uninitialized_read`` of the shared ``shared[0]``
    word read by ``output[0] = shared[0]``."""

    reviews = [f for f in report.findings if f.status == "review"]
    assert reviews, report.format()
    for finding in reviews:
        assert finding.kind == "uninitialized_read", report.format()
        assert finding.details.get("space") == "shared", finding.details
        assert _span_text(source, finding.details) == "shared[0]", finding.details


def test_copy_arrival_is_not_an_explicit_commit(tmp_path):
    for options, clean in (
        ({}, False),
        ({"commit": True, "pending": 1}, False),
        ({"commit": True}, True),
        ({"commit": True, "empty_tail": True, "pending": 1}, True),
        ({"observe_barrier": True}, True),
        ({"commit": True, "overwrite_after_wait": True}, True),
    ):
        kernel, source = arrived_open_batch_case(**options)
        sync = v2.synccheck(kernel, inputs())
        report = v2.racecheck(kernel, inputs())
        if clean:
            sync.require_clean()
            report.require_clean()
            result = v2.Engine().run(v2.transpile(kernel, cache_dir=tmp_path), inputs())
            np.testing.assert_array_equal(
                result.outputs["output"], [42 if options.get("overwrite_after_wait") else 7]
            )
        else:
            # W8-5: legacy required clean; v2 reviews the genuinely
            # uninitialized shared[0] read and nothing else.
            assert sync.verdict == "review", sync.format()
            assert {(f.status, f.kind) for f in sync.findings} == {("review", "uninitialized_read")}, sync.format()
            _assert_only_shared0_uninitialized_reviews(sync, source)
            assert report.verdict == "error", report.format()
            assert any(f.details["access_pair"] in ("write_read", "read_write") for f in report.findings), (
                report.format()
            )
            assert {(f.status, f.kind) for f in report.findings} == {
                ("error", "data_race"),
                ("review", "uninitialized_read"),
            }, report.format()
            _assert_only_shared0_uninitialized_reviews(report, source)
