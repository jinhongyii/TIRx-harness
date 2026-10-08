"""v2 copy of ``tests/numsim/runtime/test_cp_async_mbarrier_visibility.py::
test_copy_arrival_retains_earlier_copy_history`` (both params) with racecheck
delta **B2**.

Warp 0 lane 0 issues ``cp.async`` (``shared[0:4] <- source``), commits it,
either waits for it (``waited``) or leaves a second copy open
(``committed-and-open``: ``shared[8:12] <- source``), then signals the
barrier with ``cp.async.mbarrier.arrive.noinc``. Warp 1 lane 0 waits on the
barrier and reads ``shared[0]`` (or the unrelated generic ``shared[4]``),
optionally first overwriting ``source[0]``.

Legacy accepted a ``mbarrier.test_wait.parity.relaxed.cta`` spin as
acquiring the copy completions, so the ``relaxed`` cases were clean. Under
racecheck delta B2 a ``.relaxed`` mbarrier wait synchronises nothing
(arrivals and completions stay parked until a later ``fence.acquire``, and
the kernel has none). Every relaxed race is therefore between warp 0's
asynchronous copy and warp 1's access after the relaxed wait:

- ``write_read`` on shared bytes ``[0, 4)``: the first ``cp.async`` write vs.
  warp 1's ``shared[0]`` read;
- with ``reuse_source``: ``read_write`` on global ``source`` bytes ``[0, 4)``,
  each copy's read (one in ``waited``, two in ``committed-and-open``) vs.
  warp 1's ``source[0] = 42``.

The ``unrelated`` cases (generic ``shared[4]`` write, never released by the
copy arrival) and every non-relaxed case keep the legacy verdicts exactly.
Synccheck stays clean everywhere, and every numeric output is unchanged
(NumSim still produces 7 in the relaxed cases). The kernel builder is copied
verbatim from the legacy file.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def copy_arrival_case(
    *,
    unrelated=False,
    relaxed=False,
    committed=False,
    waited=False,
    extra=False,
    reuse_source=False,
):
    wait = (
        """ready[0] = T.uint32(0)
            while ready[0] == 0:
                T.ptx.mbarrier.test_wait.parity.relaxed.cta.shared.b64(
                    ready[0], barrier.ptr_to([0]), T.uint32(0))"""
        if relaxed
        else """T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)"""
    )
    extra_copy = (
        """T.ptx["cp.async.ca.shared.global"](
                shared.ptr_to([8]), source.ptr_to([0]), 16)"""
        if extra
        else "T.evaluate(0)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(source: T.Buffer((4,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_shared((12,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=16)
    ready = T.alloc_local((1,), "uint32")
    if warp == 1 and lane == 0:
        shared[4] = T.uint32(0)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        if warp == 0:
            shared[4] = T.uint32(42)
            T.ptx["cp.async.ca.shared.global"](shared.ptr_to([0]), source.ptr_to([0]), 16)
            {"T.ptx.cp.async_.commit_group()" if committed else "T.evaluate(0)"}
            {"T.ptx.cp.async_.wait_group(0)" if waited else "T.evaluate(0)"}
            {extra_copy}
            T.ptx.cp.async_.mbarrier.arrive.noinc.shared.b64(barrier.ptr_to([0]))
        else:
            {wait}
            {"source[0] = T.uint32(42)" if reuse_source else "T.evaluate(0)"}
            output[0] = {"shared[4]" if unrelated else "shared[0]"}
""",
        {"T": T},
    )


def inputs():
    return {"source": np.arange(7, 11, dtype=np.uint32), "output": np.zeros(1, np.uint32)}


def _race_signature(finding):
    """(access_pair, space, [start, end), prior is an async copy, prior warp, current warp)."""

    details = finding.details
    overlap = details["overlap"]
    prior = details["prior"]
    current = details["current"]
    return (
        details["access_pair"],
        current["space"],
        overlap["byte_offset"],
        overlap["byte_end"],
        prior["operation"]["async_op"] is not None,
        prior["operation"]["global_warp_id"],
        current["operation"]["global_warp_id"],
    )


# The unrelated generic ``shared[4] = 42`` (warp 0) vs. warp 1's read: the
# legacy race, unchanged by B2.
_UNRELATED_RACE = ("write_read", "shared", 16, 20, False, 0, 1)
# B2: warp 0's cp.async write of shared[0:4] vs. warp 1's read after the relaxed wait.
_COPY_WRITE_RACE = ("write_read", "shared", 0, 4, True, 0, 1)
# B2: a cp.async read of source[0:4] vs. warp 1's ``source[0] = 42``.
_SOURCE_REUSE_RACE = ("read_write", "global", 0, 4, True, 0, 1)


@pytest.mark.parametrize("history", ["waited", "committed-and-open"])
def test_copy_arrival_retains_earlier_copy_history(history, tmp_path):
    """Replaces ``tests/numsim/runtime/test_cp_async_mbarrier_visibility.py::test_copy_arrival_retains_earlier_copy_history`` (both params).

    Non-relaxed and ``unrelated`` cases: the legacy verdicts. Relaxed cases
    that legacy required clean: racecheck delta B2, ``error`` with exactly
    the copy-vs-waiter races listed in the module docstring; the output is
    still 7.
    """

    for relaxed in (False, True):
        for unrelated, reuse_source in ((False, False), (True, False), (False, True)):
            kernel = copy_arrival_case(
                committed=True,
                relaxed=relaxed,
                waited=history == "waited",
                extra=history != "waited",
                unrelated=unrelated,
                reuse_source=reuse_source,
            )
            case = f"{history} relaxed={relaxed} unrelated={unrelated} reuse_source={reuse_source}"
            v2.synccheck(kernel, inputs()).require_clean()
            report = v2.racecheck(kernel, inputs())
            if unrelated:
                assert report.verdict == "error", report.format()
                assert any(f.details["access_pair"] == "write_read" for f in report.findings), report.format()
                assert [_race_signature(f) for f in report.findings if f.kind == "data_race"] == [_UNRELATED_RACE], (
                    case,
                    report.format(),
                )
                assert {(f.status, f.kind) for f in report.findings} == {("error", "data_race")}, report.format()
                continue
            if relaxed:
                # Racecheck delta B2: the relaxed wait synchronises nothing.
                copies = 1 if history == "waited" else 2
                expected = [_COPY_WRITE_RACE] + ([_SOURCE_REUSE_RACE] * copies if reuse_source else [])
                assert report.verdict == "error", (case, report.format())
                assert {(f.status, f.kind) for f in report.findings} == {("error", "data_race")}, report.format()
                assert sorted(_race_signature(f) for f in report.findings) == sorted(expected), (
                    case,
                    report.format(),
                )
            else:
                report.require_clean()
            result = v2.Engine().run(v2.transpile(kernel, cache_dir=tmp_path), inputs())
            np.testing.assert_array_equal(result.outputs["output"], [7])
