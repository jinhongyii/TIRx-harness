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


# ---------------------------------------------------------------------------
# test_exclusive_tmem_uses_cta_local_lifecycle_without_placement (W6 ruling (b))


# Copied from the legacy test module; the legacy comment "the model
# deliberately permits ordinary allocation while exclusive is live" is dropped
# (W6 ruling, sync-behaviour-deltas T8).
@T.prim_func
def exclusive_lifecycle(columns: T.uint32, output: T.Buffer((2,), "uint32")):
    T.func_attr({"tirx.cuda_arch": "sm_107a"})
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.thread_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__1.exclusive.sync.aligned.shared__cta.b32(
        T.address_of(address[0]),
        columns,
    )
    if columns == 96:
        ordinary = T.alloc_buffer((1,), "uint32", scope="shared")
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(ordinary[0]),
            32,
        )
        T.ptx.tcgen05.dealloc.cta_group__1.exclusive.sync.aligned.b32(ordinary[0], 32)
    value = T.alloc_local((1,), "uint32")
    value[0] = T.Cast("uint32", cta + 1)
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    if lane == 0:
        output[cta] = value[0]
    T.ptx.tcgen05.dealloc.cta_group__1.exclusive.sync.aligned.b32(address[0], columns)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


def _exclusive(columns: int):
    return exclusive_lifecycle.specialize({exclusive_lifecycle.params[0]: columns})


def test_exclusive_tmem_576_column_lifecycle_is_clean():
    """Port of ``tests/numsim/runtime/test_launch_resource_facts.py::test_exclusive_tmem_uses_cta_local_lifecycle_without_placement`` (columns=576).

    Exclusive alloc, ld/st, exclusive dealloc, relinquish on sm_107a (576-column
    exclusive limit, sync delta T4 / W2-18): clean, outputs [1, 2]."""

    kernel = _exclusive(576)
    inputs = {"output": np.full(2, 99, dtype=np.uint32)}
    result = v2.Engine().run(v2.transpile(kernel), dict(inputs))
    np.testing.assert_array_equal(result.outputs["output"], [1, 2])
    for check in (v2.synccheck, v2.racecheck):
        report = check(kernel, dict(inputs))
        assert report.verdict == "clean", report.format()


def test_ordinary_alloc_while_exclusive_is_live_is_an_error():
    """Port of the columns=96 case of ``tests/numsim/runtime/test_launch_resource_facts.py::test_exclusive_tmem_uses_cta_local_lifecycle_without_placement``.

    W6 ruling (b), sync-behaviour-deltas T8 (PTX 9.7.18.7.1, sync-isa-answers
    Q6): an ordinary ``tcgen05.alloc`` by a CTA holding a live ``.exclusive``
    allocation is illegal (legacy accepted it). The engine run raises a protocol
    error; synccheck and racecheck report ``error``; the synccheck finding kind
    is ``tcgen_alloc_while_exclusive``."""

    kernel = _exclusive(96)
    inputs = {"output": np.full(2, 99, dtype=np.uint32)}
    with pytest.raises(v2.ExecutionError):
        v2.Engine().run(v2.transpile(kernel), dict(inputs))
    sync = v2.synccheck(kernel, dict(inputs))
    assert sync.verdict == "error", sync.format()
    assert "tcgen_alloc_while_exclusive" in {f.kind for f in sync.findings}, sync.format()
    race = v2.racecheck(kernel, dict(inputs))
    assert race.verdict == "error", race.format()
