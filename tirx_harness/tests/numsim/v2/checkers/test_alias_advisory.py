"""Logical-buffer identity over pooled shared memory and TMEM (``alias_stale_read``).

Replaces the seven ``gap_unportable`` rows of
``tests/analysis_tools/racecheck/test_native_alias_advisory.py``. The contract
``Access`` carries no logical-buffer name, so these only exist at kernel level:
lowering must give every view/sub-view one logical identity, and the review
advisory needs the site->buffer table.

``alias_stale_read`` is ruled by racecheck-behaviour-deltas P7/T8. Identity
rule W5-9: a view with the root's dtype shares the root's identity, so two
same-dtype ``decl_buffer`` names over one pool word are one logical name, and
the two legacy stale-name kernels are clean in v2. The
``keeps_one_logical_identity`` controls expect a clean verdict too.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tirx_harness.numsim import v2

from ._runnable import assert_clean, assert_no_incomplete, kinds_of, requires_v2_engine, v2_gap

pytestmark = requires_v2_engine

_TMEM_NUMERIC_GAP = v2_gap(
    "numerics fixed (V2C-24); racecheck reports alias_stale_read for a TMEM view of the same "
    "bytes (view gets its own logical identity): CONTRACT_REQUESTS W12-gaps 5 (W1)"
)

_TMEM_LAYOUT = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])
_TMEM_WIDE_LAYOUT = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def native_pool_alias_provenance(mode: T.int32, output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.alloc_buffer((1,), "int32", scope="shared")
    A_shared = T.decl_buffer((1,), "int32", data=pool.data, scope="shared")
    B_shared = T.decl_buffer((1,), "int32", data=pool.data, scope="shared")

    if lane == 0:
        A_shared[0] = 1
        if mode == 0:
            B_shared[0] = 2
            output[0] = A_shared[0]
        elif mode == 1:
            B_shared[0] = 3
            output[0] = B_shared[0]
        else:
            output[0] = A_shared[0]
            B_shared[0] = 4
            output[0] = B_shared[0]


@T.prim_func
def native_raw_ptx_pool_alias_provenance(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.alloc_buffer((1,), "uint32", scope="shared")
    A_shared = T.decl_buffer((1,), "uint32", data=pool.data, scope="shared")
    B_shared = T.decl_buffer((1,), "uint32", data=pool.data, scope="shared")
    loaded = T.alloc_local((1,), "uint32")

    if lane == 0:
        T.ptx.st.shared.u32(A_shared.ptr_to([0]), T.uint32(1))
        T.ptx.st.shared.u32(B_shared.ptr_to([0]), T.uint32(2))
        T.ptx.ld.shared.u32(loaded[0], A_shared.ptr_to([0]))
        output[0] = loaded[0]


@T.prim_func
def native_explicit_view_provenance(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((4,), "int32", scope="shared")
    matrix = storage.view(2, 2)
    transposed = matrix.rearrange("row col -> col row")

    if lane == 0:
        storage[2] = 7
        output[0] = transposed[0, 1]


@T.prim_func
def native_tmem_view_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)
    tmem_view = tmem.rearrange("(group row) col -> (group row) col", group=2)

    tmem[lane, 0] = T.cast(lane + 1, "uint32")
    output[lane] = tmem_view[lane, 0]


@T.prim_func
def native_tmem_partitioned_view_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tmem = T.decl_buffer(
        (128, 8), "uint32", scope="tmem", layout=_TMEM_WIDE_LAYOUT, allocated_addr=0
    )
    tmem_lo = tmem.sub[:, :4]
    _tmem_hi = tmem.sub[:, 4:]
    tmem_view = tmem.rearrange("(group row) col -> (group row) col", group=2)

    tmem_lo[lane, 0] = T.cast(lane + 1, "uint32")
    output[lane] = tmem_view[lane, 0]


@T.prim_func
def native_tmem_full_extent_subview_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tmem = T.decl_buffer((2, 64, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)
    tmem_view = tmem.sub[:, :, 0:4]

    tmem[0, lane, 0] = T.cast(lane + 1, "uint32")
    output[lane] = tmem_view[0, lane, 0]


@T.prim_func
def native_tmem_reused_lifetime_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    first = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)
    second = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)

    first[lane, 0] = T.cast(lane + 1, "uint32")
    second[lane, 0] = T.cast(lane + 2, "uint32")
    output[lane] = first[lane, 0]


def _assert_alias_review(report) -> None:
    assert report.verdict == "review", report.format()
    assert not [f for f in report.findings if f.status == "error"], report.format()
    assert_no_incomplete(report)
    assert "alias_stale_read" in kinds_of(report), report.format()


def _assert_numeric_then_clean(kernel) -> None:
    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    numeric = v2.Engine().run(v2.transpile(kernel), dict(inputs), outputs=("output",))
    np.testing.assert_array_equal(numeric.outputs["output"], np.arange(1, 33, dtype=np.uint32))
    report = v2.racecheck(kernel, {"output": np.zeros(32, dtype=np.uint32)})
    assert_clean(report)


def test_stale_logical_name_read_is_review():
    """Replaces ``tests/analysis_tools/racecheck/test_native_alias_advisory.py::test_public_native_stale_logical_name_read_is_review``.

    Write A_shared, write B_shared (same pool word), read A_shared. Delta
    (racecheck-behaviour-deltas P7, identity rule W5-9): ``A_shared`` and
    ``B_shared`` are int32 views of the int32 ``pool``, so both share the
    pool's logical identity and there is no stale name: clean (legacy reported
    a ``review`` ``alias_stale_read``). Only a dtype-changing view is a new name.
    """

    report = v2.racecheck(
        native_pool_alias_provenance,
        {"mode": np.int32(0), "output": np.zeros(1, dtype=np.int32)},
    )
    assert_clean(report)


def test_raw_ptx_shared_address_retains_logical_alias_owner():
    """Replaces ``tests/analysis_tools/racecheck/test_native_alias_advisory.py::test_raw_ptx_shared_address_retains_logical_alias_owner``.

    Raw ``st.shared``/``ld.shared`` through ``ptr_to`` of two pool aliases.
    Delta (racecheck-behaviour-deltas P7, identity rule W5-9): both aliases
    have the pool's dtype, so they are one logical identity: clean (legacy
    reported a ``review`` ``alias_stale_read``).
    """

    report = v2.racecheck(native_raw_ptx_pool_alias_provenance, {"output": np.zeros(1, dtype=np.uint32)})
    assert_clean(report)


def test_explicit_view_keeps_one_logical_identity():
    """Replaces ``tests/analysis_tools/racecheck/test_native_alias_advisory.py::test_public_native_explicit_view_keeps_one_logical_identity``.

    ``view`` + ``rearrange`` of one shared buffer is the same logical buffer: clean.
    """

    assert_clean(v2.racecheck(native_explicit_view_provenance, {"output": np.zeros(1, dtype=np.int32)}))


@_TMEM_NUMERIC_GAP
def test_tmem_view_keeps_one_logical_identity():
    """Replaces ``tests/analysis_tools/racecheck/test_native_alias_advisory.py::test_public_native_tmem_view_keeps_one_logical_identity``.

    NumSim output ``1..32`` through a TMEM rearrange view, and a clean racecheck.
    """

    _assert_numeric_then_clean(native_tmem_view_provenance)


@_TMEM_NUMERIC_GAP
def test_partitioned_tmem_views_keep_one_logical_identity():
    """Replaces ``tests/analysis_tools/racecheck/test_native_alias_advisory.py::test_public_native_partitioned_tmem_views_keep_one_logical_identity``."""

    _assert_numeric_then_clean(native_tmem_partitioned_view_provenance)


@_TMEM_NUMERIC_GAP
def test_full_extent_tmem_subview_keeps_one_logical_identity():
    """Replaces ``tests/analysis_tools/racecheck/test_native_alias_advisory.py::test_public_native_full_extent_tmem_subview_keeps_one_logical_identity``."""

    _assert_numeric_then_clean(native_tmem_full_extent_subview_provenance)


def test_reused_tmem_lifetime_remains_distinct():
    """Replaces ``tests/analysis_tools/racecheck/test_native_alias_advisory.py::test_public_native_reused_tmem_lifetime_remains_distinct``.

    Two ``decl_buffer``s at TMEM address 0: reading ``first`` after writing
    ``second`` is review ``alias_stale_read``.
    """

    report = v2.racecheck(native_tmem_reused_lifetime_provenance, {"output": np.zeros(32, dtype=np.uint32)})
    _assert_alias_review(report)
