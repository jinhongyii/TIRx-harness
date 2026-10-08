"""v2 copy of ``tests/numsim/runtime/test_launch_resource_facts.py::test_pointer_bits_preserve_binding_address_and_subview_offset``.

Ruling (W2, V2C-35 reversal, d5b0f09): v2 does not reproduce host pointer
bits. Per ``numsim-core/src/arena.rs`` ``addr`` (module doc): every global
allocation gets a synthetic, cudaMalloc-like base at ``GLOBAL_VA_BASE``
aligned to ``addr::GLOBAL_ALIGN`` (4 KiB); only a view *inside* a bound
region keeps its byte offset from that region's base. The legacy expectation
"address low 8 bits == host pointer low 8 bits" is therefore dropped. The
sub-view offset assertion is kept by binding the ``storage`` region too, and
addresses are cross-checked with ``Engine.address_of``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

GLOBAL_ALIGN = 1 << 12  # numsim-core arena::addr::GLOBAL_ALIGN


@T.prim_func
def observe_view_offset(
    storage: T.Buffer((320,), "uint8"),
    source: T.Buffer((32,), "uint8"),
    output: T.Buffer((32,), "uint64"),
):
    lane = T.thread_id([32])
    output[lane] = T.reinterpret("uint64", source.ptr_to([lane])) - T.reinterpret(
        "uint64", storage.ptr_to([0])
    )


# Copied verbatim from the legacy test module.
@T.prim_func
def observe_pointer_address(source: T.Buffer((32,), "uint8"), output: T.Buffer((32,), "uint64")):
    lane = T.thread_id([32])
    address = T.reinterpret("uint64", source.ptr_to([lane]))
    output[lane] = address & T.uint64(255)


@pytest.mark.parametrize("offset", [0, 1, 31, 200])
def test_subview_keeps_its_offset_inside_the_bound_region(offset):
    """The legacy sub-view assertion: ``source = storage[offset:offset+32]``
    is addressed ``offset`` bytes past ``storage`` when both are bound."""

    storage = np.zeros(320, dtype=np.uint8)
    module = v2.transpile(observe_view_offset)
    inputs = {
        "storage": storage,
        "source": storage[offset : offset + 32],
        "output": np.zeros(32, dtype=np.uint64),
    }
    engine = v2.Engine()
    result = engine.run(module, inputs)

    np.testing.assert_array_equal(
        result.outputs["output"], np.uint64(offset) + np.arange(32, dtype=np.uint64)
    )
    base = engine.address_of(module, inputs, "storage")
    assert base % GLOBAL_ALIGN == 0
    assert engine.address_of(module, inputs, "source") - base == offset


@pytest.mark.parametrize("offset", [0, 1, 31])
def test_lone_binding_gets_an_aligned_base(offset):
    """Replaces the legacy host-bits expectation: a view bound on its own is
    its own region, so its base is ``GLOBAL_ALIGN``-aligned regardless of the
    host pointer, and the kernel sees ``base + lane``."""

    storage = np.zeros(320, dtype=np.uint8)
    module = v2.transpile(observe_pointer_address)
    inputs = {"source": storage[offset : offset + 32], "output": np.zeros(32, dtype=np.uint64)}
    engine = v2.Engine()
    result = engine.run(module, inputs)

    base = engine.address_of(module, inputs, "source")
    assert base % GLOBAL_ALIGN == 0
    expected = (np.uint64(base) + np.arange(32, dtype=np.uint64)) & np.uint64(255)
    np.testing.assert_array_equal(result.outputs["output"], expected)
