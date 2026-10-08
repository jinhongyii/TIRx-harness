"""v2 ports of ``tests/numsim/runtime/test_cache_hint_ops.py``
(``test_async_applypriority_bulk_groups_are_visible_to_checkers``,
``test_valid_address_prefetch_checks_only_predicate_selected_lanes``).

Kernels are copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


@T.prim_func
def raw_cache_hint_family(
    source: T.Buffer((256,), "uint8"),
    input_map: T.TensorMap(),
    output: T.Buffer((32,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = lane < 16

    T.ptx["prefetch.L1::32B.valid_addr"](source.ptr_to([0]), pred=issue)
    T.ptx["prefetchu.L1"](source.ptr_to([0]), pred=issue)
    T.ptx["applypriority.L2::evict_normal"](source.ptr_to([0]), pred=issue)
    T.ptx["cp.async.bulk.prefetch.L2.global.L2::evict_last"](
        source.ptr_to([16]), T.uint32(32), pred=issue
    )
    T.ptx["applypriority.async.bulk.bulk_group.L2::evict_normal"](
        source.ptr_to([0]), T.uint32(32), pred=issue
    )
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.L2::evict_last"](
        T.address_of(input_map), T.int32(0), T.int32(0), pred=issue
    )
    T.ptx["applypriority.async.bulk.tensor.2d.bulk_group.L2::evict_normal"](
        T.address_of(input_map), T.int32(0), T.int32(0), pred=issue
    )
    T.ptx.cp.async_.bulk.commit_group()
    T.ptx.cp.async_.bulk.wait_group.read(0)
    output[lane] = source[lane + 32]


@T.prim_func
def predicated_valid_address(source: T.Buffer((256,), "uint8"), enabled: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    T.ptx["prefetch.L1::32B.valid_addr"](source.ptr_to([256]), pred=enabled)


def _tensor_map(array: np.ndarray) -> np.ndarray:
    return TensorMap(
        base=array,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 1),
        element_strides=(1, 1),
    ).numpy()


def _cache_hint_inputs() -> dict[str, np.ndarray]:
    return {
        "source": np.arange(256, dtype=np.uint8) ^ np.uint8(0xA5),
        "input_map": _tensor_map(np.arange(16, dtype=np.float32).reshape(4, 4)),
        "output": np.zeros(32, dtype=np.uint8),
    }


@pytest.mark.parametrize("checker", ["synccheck", "racecheck"])
def test_async_applypriority_bulk_groups_are_visible_to_checkers(checker):
    """Port of ``tests/numsim/runtime/test_cache_hint_ops.py::test_async_applypriority_bulk_groups_are_visible_to_checkers``.

    The private ``_run_synccheck``/``_run_racecheck`` become the public
    ``v2.synccheck``/``v2.racecheck``; both must be clean. Dropped pin:
    ``native_payload["stats"]["completion_operation_count"] == 64`` (a
    legacy stats count; ``applypriority.async.bulk*`` with ``bulk_group`` is
    an OpLib no-op in v2, numsim-behaviour-deltas.md row P7).
    """

    report = getattr(v2, checker)(raw_cache_hint_family, _cache_hint_inputs())
    report.require_clean()


def test_valid_address_prefetch_checks_only_predicate_selected_lanes():
    """Port of ``tests/numsim/runtime/test_cache_hint_ops.py::test_valid_address_prefetch_checks_only_predicate_selected_lanes``.

    Dropped pin: the legacy message ``"out-of-bounds"``; the port asserts the
    run with the predicate on stops with the v2 error kind ``bad_address``.
    """

    module = v2.transpile(predicated_valid_address)
    result = v2.Engine().run(module, {"source": np.zeros(256, dtype=np.uint8), "enabled": 0})
    assert result.status["kind"] == "completed"

    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"source": np.zeros(256, dtype=np.uint8), "enabled": 1})
    stops = [d for d in caught.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and (stops[0]["status"], stops[0]["kind"]) == ("error", "bad_address"), stops
