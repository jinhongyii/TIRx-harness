"""v2 delta copy of ``tests/numsim/runtime/test_bulk_reduce_s2g_f32.py::test_bulk_reduce_is_atomic_captures_source_and_publishes_only_at_full_wait``.

Triage class ``delta``. The kernel issues two ``cp.reduce.async.bulk ... add.f32``
into ``destination``, waits only with ``cp.async.bulk.wait_group.read 0``, reads
``destination[0]`` into ``observed[0]``, overwrites the shared source, then
waits with ``wait_group 0`` and reads ``observed[1]``.

Legacy pinned ``observed[0] == 10`` (destination unpublished until the full
wait). The rulings make that read a race, so its value is unspecified:
``sync-behaviour-deltas.md`` row A2 (``wait_group.read`` "acquires
``ReadsDone`` only. Never publishes destination writes. A destination read
after only ``.read`` is a race.") and ``racecheck-behaviour-deltas.md`` row X6.
v2 NumSim returns 16 (both reductions applied) and v2 racecheck reports the
two ``write_read`` races at the ``observed[0]`` read. The copy keeps every
non-racy assertion (final destination, ``observed[1] == 16``: the source is
captured before the overwrite, clean NumSim with no diagnostics, clean
synccheck) and asserts ``observed[0]`` is one of the values an interleaving of
the two reductions can give plus the racecheck findings.

Dropped pin: ``call_op_names`` (legacy ``analyze``); the copy checks the parsed
``T.ptx.cp(..., "reduce", ...)`` call in ``kernel.script()``.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


_BULK_REDUCE_ADD_F32 = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"


@T.prim_func
def bulk_reduce_add_f32_read_then_full_wait(
    source: T.Buffer((8,), "float32"),
    destination: T.Buffer((4,), "float32"),
    observed: T.Buffer((2,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared", align=16)
    if lane == 0:
        for index in T.serial(8):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx[_BULK_REDUCE_ADD_F32](destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16))
        T.ptx[_BULK_REDUCE_ADD_F32](destination.ptr_to([0]), shared.ptr_to([4]), T.uint32(16))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        observed[0] = destination[0]
        for index in T.serial(8):
            shared[index] = T.float32(1000)
        T.ptx.cp.async_.bulk.wait_group(0)
        observed[1] = destination[0]


def _inputs():
    return {
        "source": np.arange(1, 9, dtype=np.float32),
        "destination": np.full(4, np.float32(10), dtype=np.float32),
        "observed": np.zeros(2, np.float32),
    }


def test_bulk_reduce_is_atomic_captures_source_and_publishes_only_at_full_wait():
    """Delta copy (sync-behaviour-deltas A2, racecheck-behaviour-deltas X6); see the module docstring."""
    kernel = bulk_reduce_add_f32_read_then_full_wait
    inputs = _inputs()
    source = inputs["source"].copy()
    result = v2.Engine().run(v2.transpile(kernel), inputs)

    np.testing.assert_array_equal(result.outputs["destination"], np.float32(10) + source[:4] + source[4:])
    assert float(result.outputs["observed"][1]) == 16.0
    # Racy read after only `.read`: any interleaving of the two reductions.
    assert float(result.outputs["observed"][0]) in {10.0, 11.0, 15.0, 16.0}
    assert result.verdict == "clean"
    assert result.diagnostics == []

    assert v2.synccheck(kernel, _inputs()).verdict == "clean"
    report = v2.racecheck(kernel, _inputs())
    assert report.verdict == "error", report.format()
    assert [(f.status, f.kind) for f in report.findings] == [("error", "data_race")] * 2
    assert all(f.details["access_pair"] == "write_read" for f in report.findings)

    script = kernel.script()
    assert '"reduce", "async", "bulk"' in script and '"add"' in script
