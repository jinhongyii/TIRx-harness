"""v2 ports of the legacy packed-f32x2 checker tests; the private legacy
``checkers._run_racecheck`` / ``_run_synccheck`` entry points (and their
``cache_dir`` / ``max_workers`` knobs) become public ``v2.racecheck`` /
``v2.synccheck``. Kernels copied verbatim from
``tests/numsim/runtime/test_packed_f32x2_float32_destination.py``."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def packed_f32x2_float32_destination(output: T.Buffer((32, 4, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pair = T.alloc_buffer((2,), "float32", scope="local")
    pair_u64 = pair.view("uint64")

    lhs: T.let = T.cuda.make_float2(T.cast(lane, "float32"), T.cast(lane + 1, "float32"))
    rhs: T.let = T.cuda.make_float2(T.float32(0.75), T.float32(0.5))
    addend: T.let = T.cuda.make_float2(T.float32(1), T.float32(2))

    T.ptx.fma.rn.ftz.f32x2(pair_u64[0], lhs, rhs, addend)
    output[lane, 0, 0] = pair[0]
    output[lane, 0, 1] = pair[1]

    T.ptx.add.rn.ftz.f32x2(pair_u64[0], lhs, rhs)
    output[lane, 1, 0] = pair[0]
    output[lane, 1, 1] = pair[1]

    T.ptx.sub.rn.ftz.f32x2(pair_u64[0], lhs, rhs)
    output[lane, 2, 0] = pair[0]
    output[lane, 2, 1] = pair[1]

    T.ptx.mul.rn.ftz.f32x2(pair_u64[0], lhs, rhs)
    output[lane, 3, 0] = pair[0]
    output[lane, 3, 1] = pair[1]


@T.prim_func
def packed_store_races_with_second_element(output: T.Buffer((2,), "float32")):
    """The packed store spans smem[0..2); warp 1 concurrently writes smem[1]."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    smem = T.alloc_buffer((2,), "float32", scope="shared")
    smem_pair = smem.view("uint64")

    lhs: T.let = T.cuda.make_float2(T.float32(1), T.float32(2))
    rhs: T.let = T.cuda.make_float2(T.float32(3), T.float32(4))

    if warp == 0 and lane == 0:
        T.ptx.add.rn.ftz.f32x2(smem_pair[0], lhs, rhs)
    if warp == 1 and lane == 0:
        smem[1] = T.float32(7)
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        output[0] = smem[0]
        output[1] = smem[1]


def test_checkers_accept_the_float32_pair_destination():
    """Port of ``tests/numsim/runtime/test_packed_f32x2_float32_destination.py::test_checkers_accept_the_float32_pair_destination``."""

    inputs = {"output": np.zeros((32, 4, 2), dtype=np.float32)}

    sync = v2.synccheck(packed_f32x2_float32_destination, inputs)
    assert sync.verdict == "clean", sync.format()

    race = v2.racecheck(packed_f32x2_float32_destination, inputs)
    assert race.verdict == "clean", race.format()


def test_packed_store_footprint_covers_the_second_element():
    """Port of ``tests/numsim/runtime/test_packed_f32x2_float32_destination.py::test_packed_store_footprint_covers_the_second_element``.

    Positive control: the packed f32x2 write through the uint64 view covers
    both float32 elements, so warp 1's write of the second element races.

    v2 ``RaceReport.findings`` also lists ``review`` advisories: the final
    ``smem[0]`` read through the other logical name of the bytes the packed
    store wrote through ``smem_pair`` is the ``alias_stale_read`` review
    (racecheck-behaviour-deltas.md P7/T8). The legacy error-finding list is
    asserted on the ``error`` findings.
    """

    report = v2.racecheck(packed_store_races_with_second_element, {"output": np.zeros(2, dtype=np.float32)})

    assert report.verdict == "error", report.verdict
    errors = [f for f in report.findings if f.status == "error"]
    assert [f.details["access_pair"] for f in errors] == ["write_write"], report.format()
    assert {f.kind for f in report.findings if f.status != "error"} <= {"alias_stale_read"}, report.format()
