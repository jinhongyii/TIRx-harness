"""v2 copies of the five category-A functions in ``tests/numsim/registry`` that had no
v2 copy (W11 legacy-import batch). Their modules import the legacy frontend
(``analyze``/``verify``, ``support/manifest`` emission helpers) and are deleted
with the legacy layer; these copies keep the observable facts.

- ``test_call_resolution.py::test_fetch_register_alias_normalization_is_explicit``:
  ``mov_sreg(32, "laneid")`` and ``"%laneid"`` lower to the same Program
  (legacy compared the emitted Rust modules).
- ``test_ptx_spdecompress_registry.py::test_spdecompress_accepts_each_b32_carrier_without_changing_static_shape``:
  every b32 carrier dtype lowers to the same ``Ptx`` op (same ``OpKey``) with the
  same operand count (legacy compared the emitted call heads).
- ``test_copy_dispatch_contract.py::test_fallback_hint_is_accepted_for_thread_owned_local_copy``:
  ``v2.transpile`` accepts the kernel.
- ``test_copy_dispatch_contract.py::test_same_warp_copy_remaps_lane_owned_values``:
  unchanged output assertion.
- ``test_copy_dispatch_contract.py::test_default_thread_owned_local_copy_uses_physical_owners``:
  delta numsim-behaviour-deltas F4. TVM dispatches the warpgroup local->local
  ``Tx.wg.copy`` to ``copy/fallback`` (scalar, single thread), whose code reads
  registers other threads own; v2 runs that code and stops with a ``trap``
  (register-layout element owned by another thread) at the copy, where legacy
  copied every thread's own elements.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.backend.cuda.ptx.table import TABLE, mods, operand_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout, laneid, wg_local_layout

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2
from tirx_harness.numsim.v2.lowering import lower

pytestmark = requires_v2_engine


# -- test_call_resolution ----------------------------------------------------


def _mov_sreg_kernel(register: str):
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.mov_sreg(32, "{register}")
''',
        {"T": T},
    )


def _code(kernel):
    program = lower(kernel)
    return [str(instr) for instr in program.code], [r.to_json() for r in program.regs]


@v2_gap("W11-6: lowering rejects mov_sreg(32, '%laneid') as 'special register %%laneid' instead of normalizing the % prefix")
def test_fetch_register_alias_normalization_is_explicit():
    """Copy; ``laneid`` and ``%laneid`` lower to identical code and registers."""
    plain, prefixed = _code(_mov_sreg_kernel("laneid")), _code(_mov_sreg_kernel("%laneid"))
    assert plain == prefixed
    assert any("LaneId" in instr for instr in plain[0])


# -- test_ptx_spdecompress_registry ------------------------------------------


def _spdecompress_lanes(spelling: str) -> dict[str, int]:
    entry = TABLE["spdecompress"]
    modifiers = mods(entry, spelling.removeprefix("spdecompress.").split("."))
    assert entry.check(modifiers) is None
    return {operand.name: count for operand, _offset, count in operand_layout(entry, modifiers)}


def _spdecompress_kernel(spelling: str, dtypes: tuple[str, ...]):
    declarations = "\n".join(
        f'    operand_{index} = T.local_scalar("{dtype}")' for index, dtype in enumerate(dtypes)
    )
    arguments = ", ".join(f"operand_{index}" for index in range(len(dtypes)))
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel():
    T.device_entry()
{declarations}
    T.ptx["{spelling}"]({arguments})
''',
        {"T": T},
    )


def test_spdecompress_accepts_each_b32_carrier_without_changing_static_shape():
    """Copy; the carrier dtype does not change the lowered op or its operand shape."""
    spelling = "spdecompress.b8.b4.sp::2:4.x2"
    lanes = _spdecompress_lanes(spelling)
    plain = ("uint32",) * (lanes["data"] + lanes["mdata"] + lanes["cdata"])
    mixed = ("uint32",) * lanes["data"] + ("int32",) * lanes["mdata"] + ("float32",) * lanes["cdata"]

    def ptx_ops(kernel):
        program = lower(kernel)
        return [
            (program.ops[i.op], len(i.dsts), len(i.srcs))
            for i in program.code
            if i.variant == "Ptx" and program.ops[i.op].name == "tirx.ptx.spdecompress"
        ]

    ops = ptx_ops(_spdecompress_kernel(spelling, mixed))
    assert len(ops) == 1
    assert ops == ptx_ops(_spdecompress_kernel(spelling, plain))


# -- test_copy_dispatch_contract ---------------------------------------------


@T.prim_func
def _thread_owned_local_copy(
    source: T.Buffer((128, 2), "float32"), output: T.Buffer((128, 2), "float32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    source_storage: T.f32[2]
    destination_storage: T.f32[2]
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=wg_local_layout(2))
    for column in T.serial(2):
        source_view[thread, column] = source[thread, column]
    Tx.wg.copy(destination_view, source_view)
    for column in T.serial(2):
        output[thread, column] = destination_view[thread, column]


@T.prim_func
def _same_warp_owner_remap(
    source: T.Buffer((32, 32), "float32"), output: T.Buffer((32, 32), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_storage: T.f32[32]
    destination_storage: T.f32[32]
    source_view = source_storage.view(32, 32, layout=TileLayout(S[(32, 32) : (1 @ laneid, 1)]))
    destination_view = destination_storage.view(
        32, 32, layout=TileLayout(S[(32, 32) : (1, 1 @ laneid)])
    )
    for column in T.serial(32):
        source_view[lane, column] = source[lane, column]
    Tx.warp.copy(destination_view, source_view)
    for row in T.serial(32):
        output[row, lane] = destination_view[row, lane]


@T.prim_func
def _invalid_forced_thread_owned_local_copy():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage: T.f32[2]
    destination_storage: T.f32[2]
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=wg_local_layout(2))
    Tx.wg.copy(destination_view, source_view, dispatch="fallback")


def test_fallback_hint_is_accepted_for_thread_owned_local_copy():
    """Copy; transpile-only, as legacy."""
    v2.transpile(_invalid_forced_thread_owned_local_copy)


def test_same_warp_copy_remaps_lane_owned_values():
    """Copy; output assertion unchanged."""
    source = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) + np.float32(0.25)
    result = v2.Engine().run(
        v2.transpile(_same_warp_owner_remap), {"source": source, "output": np.zeros_like(source)}
    )
    np.testing.assert_array_equal(result.outputs["output"], source)


def test_default_thread_owned_local_copy_uses_physical_owners():
    """Delta copy (F4): the dispatched single-thread fallback reads other threads'
    registers, so v2 stops with a ``trap`` at the copy instead of copying."""
    source = np.arange(128 * 2, dtype=np.float32).reshape(128, 2) + np.float32(0.25)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(
            v2.transpile(_thread_owned_local_copy), {"source": source, "output": np.zeros_like(source)}
        )
    stops = [d for d in caught.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and (stops[0]["status"], stops[0]["kind"]) == ("error", "trap"), caught.value.diagnostics
    # The single fallback thread (thread 0 of the warpgroup) is the faulting lane.
    assert (stops[0]["warp"], stops[0]["lanes"]) == (0, "WarpMask(0x00000001)"), stops[0]


@v2_gap("W11-5: the register-owner Assert lowered for a dispatched tile op has no site, so the trap is not anchored at Tx.wg.copy")
def test_default_thread_owned_local_copy_trap_is_anchored_at_the_copy():
    source = np.arange(128 * 2, dtype=np.float32).reshape(128, 2) + np.float32(0.25)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(
            v2.transpile(_thread_owned_local_copy), {"source": source, "output": np.zeros_like(source)}
        )
    stop = [d for d in caught.value.diagnostics if d.get("status") == "error"][0]
    span = stop["source_span"]
    with open(span["source_name"]) as handle:
        assert "Tx.wg.copy" in handle.read().splitlines()[span["line"] - 1], stop
