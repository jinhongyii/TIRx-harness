"""v2 delta copy of ``tests/numsim/integration/test_multi_kernel_artifact.py::test_pointer_words_keep_allocation_binding_across_phases``.

Triage (W9 phase 6, internal other-assertion): **delta**,
``CONTRACT_REQUESTS.md`` "W2 (2026-10-08): V2C-35 ..." item 1 (V2C-35 reversed
ruling): a top-level buffer's base is a fresh synthetic 4096-aligned engine
address and the host pointer's bits are ignored. Legacy pinned the exported
pointer words to ``source.ctypes.data + 4 * i`` and read them back from the
caller's ``pointers`` array (in-place mutation). v2 writes the engine address
(``Engine.address_of``, W8-7) and never mutates its inputs, so the copy reads
``result.outputs["k0:pointers"]``. The cross-phase load through the words
(``k1:output == source``) is unchanged.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def export_global_pointer_words(
    source: T.Buffer((32,), "uint32"), pointers: T.Buffer((32,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointers[lane] = T.reinterpret("uint64", source.ptr_to([lane]))


@T.prim_func
def consume_global_pointer_words(
    pointers: T.Buffer((32,), "uint64"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", pointers[lane]))


def test_pointer_words_keep_allocation_binding_across_phases():
    """Delta copy (V2C-35); see the module docstring."""

    source = np.arange(32, dtype=np.uint32) + 100
    pointers = np.zeros(32, dtype=np.uint64)
    module = v2.transpile([export_global_pointer_words, consume_global_pointer_words])
    inputs = {
        "k0:source": source,
        "k0:pointers": pointers,
        "k1:pointers": pointers,
        "k1:output": np.zeros(32, dtype=np.uint32),
    }
    engine = v2.Engine()
    result = engine.run(module, inputs, outputs=("k1:output", "k0:pointers"))

    np.testing.assert_array_equal(result.outputs["k1:output"], source)
    base = engine.address_of(module, inputs, "k0:source")
    assert base % 4096 == 0
    np.testing.assert_array_equal(
        result.outputs["k0:pointers"], np.uint64(base) + np.arange(32, dtype=np.uint64) * np.uint64(4)
    )
    np.testing.assert_array_equal(pointers, np.zeros(32, dtype=np.uint64))
