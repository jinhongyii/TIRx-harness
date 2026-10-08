"""Hidden accesses of tile and helper primitives (``T.cuda.cta_sum`` scratch and
internal barriers, ``Tx.warp.permute_layout`` snapshot/zero-fill, a shared
index load inside a tile-primitive region).

Replaces six ``gap_unportable`` rows of
``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py``: lowering
must emit these accesses (and barriers) for Racecheck to see them.

No spec mentions these hidden accesses (test-migration.md no-spec item 10).
The ``cta_sum`` scratch/barrier tests pass, and so do the ``permute_layout``
zero-fill tests since W12's permute port (f1e8225; numsim-behaviour-deltas F1).
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout

from tirx_harness.numsim import v2

from ._runnable import (
    RACE,
    assert_clean,
    assert_error_kind,
    assert_no_incomplete,
    race_access_pairs,
    requires_v2_engine,
)

pytestmark = [requires_v2_engine]


_RACECHECK_PERMUTED_SHARED_LAYOUT = TileLayout(S[(4, 32) : (1, 4)])


@T.prim_func
def native_racecheck_tile_region_shared_index(
    source: T.Buffer((1,), "float32"), output: T.Buffer((32,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    index = T.alloc_buffer((1,), "int32", scope="shared")
    value: T.f32[1]
    if lane == 0:
        index[0] = 0
    T.cuda.warp_sync()
    Tx.copy(value[:], source[index[0] : index[0] + 1])
    output[lane] = value[0]


@T.prim_func
def native_racecheck_cta_reduce_hidden_scratch_waw(output: T.Buffer((1,), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((2,), "float32", scope="shared")
    if (warp == 1) and (lane == 0):
        scratch[0] = T.float32(7)
    reduced: T.let = T.cuda.cta_sum(T.Cast("float32", lane), 2, scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0] = reduced


@T.prim_func
def native_racecheck_cta_reduce_scratch_clean(output: T.Buffer((1,), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((2,), "float32", scope="shared")
    reduced: T.let = T.cuda.cta_sum(T.Cast("float32", warp * 32 + lane), 2, scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0] = reduced


@T.prim_func
def native_racecheck_single_warp_cta_reduce(output: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((1,), "float32", scope="shared")
    reduced: T.let = T.cuda.cta_sum(T.Cast("float32", lane), 1, scratch.ptr_to([0]))
    if lane == 0:
        output[0] = reduced


@T.prim_func
def native_racecheck_shared_permute_snapshot_waw():
    T.device_entry()
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])
    source = T.alloc_buffer((128,), "uint32", scope="shared")
    destination = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_RACECHECK_PERMUTED_SHARED_LAYOUT
    )
    Tx.warp.permute_layout(destination[:], source[:])


@T.prim_func
def native_racecheck_shared_permute_snapshot_source_race():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    source = T.alloc_buffer((128,), "uint32", scope="shared")
    destination = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_RACECHECK_PERMUTED_SHARED_LAYOUT
    )
    if warp == 0:
        if lane == 0:
            source[0] = 1
    else:
        Tx.warp.permute_layout(destination[:], source[:])


def _assert_shared_race(report, pairs: set[str]) -> None:
    assert_no_incomplete(report)
    assert_error_kind(report, RACE)
    assert race_access_pairs(report) & pairs, report.format()


def test_observes_shared_load_hidden_in_tile_region_index():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_observes_shared_load_hidden_in_tile_region_index``.

    The ``index[0]`` load that addresses ``Tx.copy``'s source region must run
    (NumSim copies ``source[0] == 3.0`` to every lane) and Racecheck is clean
    (lane 0's write is ordered by ``warp_sync``). The legacy assertion that the
    read is attributed to the tile op's source id with 32 active lanes needs
    the access log, which the v2 report does not expose.
    """

    inputs = {"source": np.array([3.0], dtype=np.float32), "output": np.zeros(32, dtype=np.float32)}
    numeric = v2.Engine().run(v2.transpile(native_racecheck_tile_region_shared_index), dict(inputs), outputs=("output",))
    np.testing.assert_array_equal(numeric.outputs["output"], np.full(32, 3.0, dtype=np.float32))
    assert_clean(v2.racecheck(native_racecheck_tile_region_shared_index, {
        "source": np.array([3.0], dtype=np.float32), "output": np.zeros(32, dtype=np.float32),
    }))


def test_permute_snapshot_zero_fill_retains_shared_accesses():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_permute_snapshot_zero_fill_retains_shared_accesses``.

    Both warps run the same ``permute_layout`` into one shared destination:
    a shared write/write race.
    """

    _assert_shared_race(v2.racecheck(native_racecheck_shared_permute_snapshot_waw, {}), {"write_write"})


def test_permute_snapshot_records_zero_fill_source_reads():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_permute_snapshot_records_zero_fill_source_reads``.

    Warp 0 writes ``source[0]`` while warp 1's ``permute_layout`` snapshots
    ``source``: a shared read/write race.
    """

    _assert_shared_race(
        v2.racecheck(native_racecheck_shared_permute_snapshot_source_race, {}), {"read_write", "write_read"}
    )


def test_records_cta_reduce_hidden_scratch_accesses():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_records_cta_reduce_hidden_scratch_accesses``.

    Warp 1 writes ``scratch[0]`` unordered with ``cta_sum``'s hidden scratch
    writes: a shared write/write race.
    """

    _assert_shared_race(
        v2.racecheck(native_racecheck_cta_reduce_hidden_scratch_waw, {"output": np.zeros(1, dtype=np.float32)}),
        {"write_write"},
    )


def test_cta_reduce_internal_barriers_order_scratch_accesses():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_cta_reduce_internal_barriers_order_scratch_accesses``.

    ``cta_sum``'s internal barriers order its own scratch traffic: clean.
    """

    report = v2.racecheck(native_racecheck_cta_reduce_scratch_clean, {"output": np.zeros(1, dtype=np.float32)})
    assert_clean(report)


def test_single_warp_cta_reduce_is_clean_and_numeric():
    """Replaces ``tests/analysis_tools/racecheck/test_native_racecheck_artifact.py::test_native_racecheck_single_warp_cta_reduce_is_clean_and_numeric``.

    Single-warp ``cta_sum`` of lane ids is ``sum(range(32))`` and Racecheck is clean.
    """

    numeric = v2.Engine().run(
        v2.transpile(native_racecheck_single_warp_cta_reduce), {"output": np.zeros(1, dtype=np.float32)}, outputs=("output",)
    )
    assert numeric.outputs["output"][0] == np.float32(sum(range(32)))
    assert_clean(v2.racecheck(native_racecheck_single_warp_cta_reduce, {"output": np.zeros(1, dtype=np.float32)}))
