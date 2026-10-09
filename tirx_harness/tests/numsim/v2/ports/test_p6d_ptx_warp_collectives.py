"""v2 copies of two ``tests/numsim/runtime/test_ptx_warp_collectives.py`` tests.

- ``test_raw_ptx_match_redux_and_activemask_match_warp_oracles``: port. Every
  numerical assertion and the partial-mask rejection pass under v2; the only
  failure was the legacy-internal pin ``{tirx.ptx.activemask, ...} <=
  call_op_names(module.spec.kernels[0])`` (legacy ``support/manifest.py`` over
  the legacy kernel spec), which is dropped. The rejection is asserted by
  exception type and kind (``invalid_operand``), not legacy text.
- ``test_raw_ptx_shuffle_reads_only_selected_source_lanes``: delta. Each lane
  evaluates its own ``source[0]`` operand of ``shfl.sync`` (a TIR load of a
  thread-local), so the 24 lanes that never wrote ``source`` read it
  uninitialized in BOTH runs. v2 reports these per lane at the register load
  (``CONTRACT_REQUESTS.md`` V2C-19/20 and "Tracked locals are ``Space::Reg``":
  W1's ``lowering/uninit.py`` keeps a maybe-unwritten local as a per-lane
  ``Space::Reg`` buffer, and the engine reports ``uninitialized_read`` with
  ``space: reg``). Legacy reported only reads whose value a selected source
  lane consumed, so its ``diagnostics == []`` for the selected-only run no
  longer holds. Kept: both output oracles, and the review/``uninitialized_read``
  shape of the second run.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def raw_ptx_shuffle_sparse_sources(
    selector: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source = T.alloc_local((1,), "float32")
    shuffled = T.alloc_local((1,), "uint32")

    if lane % 4 == 2:
        source[0] = T.cast(lane + 100, "float32")
    T.ptx.shfl_sync.idx.b32(
        shuffled[0],
        T.reinterpret("uint32", source[0]),
        selector[lane],
        T.uint32(31),
        T.uint32(0xFFFFFFFF),
    )
    output[lane] = shuffled[0]


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


def _matching_lane_masks(values: np.ndarray) -> np.ndarray:
    bits = values.view(f"uint{values.dtype.itemsize * 8}")
    return np.array(
        [sum(1 << src for src, source in enumerate(bits) if source == value) for value in bits],
        dtype=np.uint32,
    )


def test_raw_ptx_shuffle_reads_only_selected_source_lanes():
    """Delta copy of ``tests/numsim/runtime/test_ptx_warp_collectives.py::test_raw_ptx_shuffle_reads_only_selected_source_lanes`` (V2C-19/20; see module docstring)."""

    lanes = np.arange(32, dtype=np.uint32)
    initialized_sources = (lanes // np.uint32(4)) * np.uint32(4) + np.uint32(2)
    module = v2.transpile(raw_ptx_shuffle_sparse_sources)

    selected_only = v2.Engine().run(
        module, {"selector": initialized_sources, "output": np.zeros(32, dtype=np.uint32)}
    )
    np.testing.assert_array_equal(
        selected_only.outputs["output"],
        (initialized_sources + np.uint32(100)).astype(np.float32).view(np.uint32),
    )
    for run in (
        selected_only,
        v2.Engine().run(module, {"selector": lanes, "output": np.zeros(32, dtype=np.uint32)}),
    ):
        assert len(run.diagnostics) == 24  # the lanes with lane % 4 != 2
        assert {item["status"] for item in run.diagnostics} == {"review"}
        assert {item["kind"] for item in run.diagnostics} == {"uninitialized_read"}
        assert {item["space"] for item in run.diagnostics} == {"reg"}


def test_raw_ptx_match_redux_and_activemask_match_warp_oracles():
    """Port of ``tests/numsim/runtime/test_ptx_warp_collectives.py::test_raw_ptx_match_redux_and_activemask_match_warp_oracles`` (``call_op_names`` pin dropped)."""

    lanes = np.arange(32, dtype=np.uint32)
    source32 = (lanes % np.uint32(5)) | (np.uint32(1) << (lanes % np.uint32(31)))
    source64 = (lanes % np.uint32(3)).astype(np.uint64) * np.uint64(0x100000001)
    module = v2.transpile(raw_ptx_match_redux_and_activemask)

    def inputs(mask):
        return {
            "source32": source32,
            "source64": source64,
            "partial_mask": np.array([mask], dtype=np.uint32),
            "output": np.zeros((32, 13), dtype=np.uint32),
        }

    output = v2.Engine().run(module, inputs(0xFF)).outputs["output"]
    np.testing.assert_array_equal(
        output[:, 0], np.concatenate((np.full(8, 0xFF, dtype=np.uint32), np.zeros(24, dtype=np.uint32)))
    )
    np.testing.assert_array_equal(output[:, 1], _matching_lane_masks(source32))
    np.testing.assert_array_equal(output[:, 2], _matching_lane_masks(source64))
    np.testing.assert_array_equal(output[:, 3], np.zeros(32, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 4:6], np.zeros((32, 2), dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 6], np.full(32, 0xFFFFFFFF, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 7], np.ones(32, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 8], np.full(32, np.bitwise_and.reduce(source32), dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 9], np.full(32, np.bitwise_or.reduce(source32), dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 10], np.full(32, np.bitwise_xor.reduce(source32), dtype=np.uint32))
    np.testing.assert_array_equal(output[:8, 11], _matching_lane_masks(source32[:8]))
    np.testing.assert_array_equal(output[8:, 11:], np.zeros((24, 2), dtype=np.uint32))
    np.testing.assert_array_equal(output[:8, 12], np.full(8, np.bitwise_xor.reduce(source32[:8]), dtype=np.uint32))

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, inputs(0xFFFFFFFF))
    stops = [d for d in excinfo.value.diagnostics if d.get("status") == "error"]
    assert stops and stops[0]["kind"] == "invalid_operand", excinfo.value.diagnostics
    assert "participant mask names an inactive lane" in str(excinfo.value)
