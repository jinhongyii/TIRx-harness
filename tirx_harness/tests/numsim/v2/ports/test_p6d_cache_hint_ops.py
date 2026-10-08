"""v2 port of ``tests/numsim/runtime/test_cache_hint_ops.py::test_cache_hint_family_validates_operands_without_changing_memory``.

Triage class ``port``. v2 runs the cache-hint family with the legacy data
result (``output == source[32:64]``, clean, no diagnostics, input untouched).
The legacy function failed only on implementation pins:

- ``CACHE_HINT_OPS <= call_op_names(module.spec.kernels[0])`` (legacy
  ``analyze`` source map; v2 modules have no legacy ``spec.kernels[].source_map``);
- six ``... in module.rust_source`` checks (legacy Rust codegen).

The copy instead checks that the parsed PrimFunc keeps each of the seven
cache-hint calls (``kernel.script()``), and adds clean synccheck/racecheck
verdicts. Kernel and inputs copied verbatim (``TensorMap`` from the public
package).
"""

from __future__ import annotations

import numpy as np
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


_CALLS = (
    'T.ptx.prefetch(T.address_of(source[0]), issue, "", "L1::32B", "valid_addr", "pred")',
    'T.ptx.prefetchu(T.address_of(source[0]), issue, "L1", "pred")',
    'T.ptx.applypriority(T.address_of(source[0]), issue, "", "L2::evict_normal", "pred")',
    '"async", "bulk", "prefetch", "L2", "global", "L2::evict_last", "pred")',
    '"async", "bulk", "", "bulk_group", "L2::evict_normal", "pred")',
    '"async", "bulk", "prefetch", "tensor", "2d", "L2", "global", "", "L2::evict_last", "pred")',
    '"async", "bulk", "tensor", "2d", "", "bulk_group", "", "L2::evict_normal", "pred")',
)


def test_cache_hint_family_validates_operands_without_changing_memory():
    """Port of ``tests/numsim/runtime/test_cache_hint_ops.py::test_cache_hint_family_validates_operands_without_changing_memory``
    (dropped: ``call_op_names`` and ``rust_source`` pins)."""
    inputs = _cache_hint_inputs()
    source_before = inputs["source"].copy()

    result = v2.Engine().run(v2.transpile(raw_cache_hint_family), inputs)

    np.testing.assert_array_equal(result.outputs["output"], source_before[32:64])
    np.testing.assert_array_equal(inputs["source"], source_before)
    assert result.verdict == "clean"
    assert result.diagnostics == []
    script = raw_cache_hint_family.script()
    for call in _CALLS:
        assert call in script, call
    for checker in (v2.synccheck, v2.racecheck):
        assert checker(raw_cache_hint_family, _cache_hint_inputs()).verdict == "clean"
