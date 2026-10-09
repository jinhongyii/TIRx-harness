"""v2 port of the legacy raw-PTX match/redux checker test; the private
legacy ``checkers._run_racecheck`` / ``_run_synccheck`` (and their
``cache_dir`` / ``max_workers`` knobs) become public ``v2.racecheck`` /
``v2.synccheck``. Kernel copied verbatim from
``tests/numsim/runtime/test_ptx_warp_collectives.py``."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def raw_ptx_match_redux_and_activemask(
    source32: T.Buffer((32,), "uint32"),
    source64: T.Buffer((32,), "uint64"),
    partial_mask: T.Buffer((1,), "uint32"),
    output: T.Buffer((32, 13), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full = T.uint32(0xFFFFFFFF)
    active = T.alloc_local((1,), "uint32")
    all_equal = T.alloc_local((1,), "uint32")

    if lane < 8:
        T.ptx.activemask.b32(active[0])
        output[lane, 0] = active[0]
        T.ptx.match.any.sync.b32(output[lane, 11], source32[lane], partial_mask[0])
        T.ptx.redux_sync.xor.b32(output[lane, 12], source32[lane], partial_mask[0])
    T.ptx.match.any.sync.b32(output[lane, 1], source32[lane], full)
    T.ptx.match.any.sync.b64(output[lane, 2], source64[lane], full)
    T.ptx.match.all.sync.b32(output[lane, 3], source32[lane], full)
    T.ptx.match.all.sync.b32(output[lane, 4], all_equal[0], source32[lane], full)
    output[lane, 5] = all_equal[0]
    T.ptx.match.all.sync.b64(output[lane, 6], all_equal[0], T.uint64(0x0123456789ABCDEF), full)
    output[lane, 7] = all_equal[0]
    T.ptx.redux_sync.and_.b32(output[lane, 8], source32[lane], full)
    T.ptx.redux_sync.or_.b32(output[lane, 9], source32[lane], full)
    T.ptx.redux_sync.xor.b32(output[lane, 10], source32[lane], full)


@pytest.mark.parametrize("checker", [v2.synccheck, v2.racecheck], ids=["synccheck", "racecheck"])
def test_raw_ptx_match_collectives_execute_in_checkers(checker):
    """Port of ``tests/numsim/runtime/test_ptx_warp_collectives.py::test_raw_ptx_match_collectives_execute_in_checkers``."""

    checker(
        raw_ptx_match_redux_and_activemask,
        {
            "source32": np.arange(32, dtype=np.uint32),
            "source64": np.arange(32, dtype=np.uint64),
            "partial_mask": np.array([0xFF], dtype=np.uint32),
            "output": np.zeros((32, 13), dtype=np.uint32),
        },
    ).require_clean()
