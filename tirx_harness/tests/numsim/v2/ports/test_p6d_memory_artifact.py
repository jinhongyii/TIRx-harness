"""v2 delta copy of ``tests/numsim/integration/test_memory_artifact.py::test_bulk_read_wait_does_not_publish_destination_before_full_wait``.

Triage (W9 phase 6, internal other-assertion): **delta**,
``sync-behaviour-deltas.md`` A2 and ``racecheck-behaviour-deltas.md`` X6 (PTX
§9.7.10.28.6.2): ``cp.async.bulk.wait_group.read`` only acquires the
source-read milestone, so a destination read after it alone is a data race
and its value is unspecified. Legacy pinned ``observed[0] == 0x17`` (the
pre-copy byte); v2 NumSim returns ``0x40`` (the copied byte) for seeds 0-5,
and ``v2.racecheck`` reports the read as a ``write_read`` ``data_race``. Same
ruling as the ``test_bulk_reduce_s2g_f32`` copy.

Kept: the destination equals the source, ``observed[1]`` (after the full
wait) is the copied byte, synccheck clean. ``observed[0]`` is asserted to be
one of the two legal values.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def bulk_s2g_read_then_full_wait(
    source: T.Buffer((16,), "uint8"),
    destination: T.Buffer((16,), "uint8"),
    observed: T.Buffer((2,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane == 0:
        for index in T.serial(16):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.uint64(0x1000000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        observed[0] = destination[0]
        T.ptx.cp.async_.bulk.wait_group(0)
        observed[1] = destination[0]


def _inputs(source):
    return {
        "source": source,
        "destination": np.full(16, np.uint8(0x17), dtype=np.uint8),
        "observed": np.zeros(2, dtype=np.uint8),
    }


def test_bulk_read_wait_does_not_publish_destination_before_full_wait():
    """Delta copy (rows A2 / X6); see the module docstring."""

    source = np.arange(16, dtype=np.uint8) + np.uint8(0x40)
    module = v2.transpile(bulk_s2g_read_then_full_wait)
    for seed in range(4):
        result = v2.Engine(seed=seed).run(module, _inputs(source))
        np.testing.assert_array_equal(result.outputs["destination"], source)
        assert result.outputs["observed"][1] == source[0]
        assert result.outputs["observed"][0] in (0x17, source[0])

    race = v2.racecheck(bulk_s2g_read_then_full_wait, _inputs(source))
    assert race.verdict == "error", race.format()
    assert {(f.status, f.kind) for f in race.findings} == {("error", "data_race")}
    assert v2.synccheck(bulk_s2g_read_then_full_wait, _inputs(source)).verdict == "clean"
