"""v2 ports of legacy ``tests/numsim/runtime/test_tcgen_codegen.py`` tests.

The legacy tests pinned frontend internals (``analyze`` source map,
``decode_ptx_call`` payloads), ``module.rust_source`` text, run stats and
error message text. The copies assert the same facts through the v2 module
(``CompiledModule.spec`` sites and the Program document) and the observable
run outcome (outputs, error kind). Kernels are copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.support.kernels import tcgen_lifecycle_two_cta
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def tcgen_control_calls():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2,), "uint64", scope="shared")
    address = shared.view("uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 128)
    if lane == 0:
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(shared[1])
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(shared[1]), T.uint16(3), pred=T.uint32(1)
        )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    T.ptx.setmaxnreg.inc.sync.aligned.u32(128)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), 128)


@T.prim_func
def tcgen_fence_noop_forms(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.warp_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    output[lane] = T.cast(lane + 1, "uint32")


@T.prim_func
def tcgen_runtime_divergent_dealloc(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    runtime_address = T.alloc_local((1,), "uint32")
    runtime_address[0] = lane
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(runtime_address[0], 64)
    if lane == 0:
        output[0] = T.uint32(1)


@T.prim_func
def tmem_pool_non_power_of_two_columns(output: T.Buffer((1,), "int32")):
    T.device_entry()
    T.cta_id([1])
    T.warp_id([1])
    thread_id = T.thread_id([32])

    pool = T.SMEMPool()
    tmem_address = pool.alloc((1,), "uint32", align=4)
    tmem_pool = T.TMEMPool(
        pool,
        total_cols=160,
        cta_group=1,
        tmem_addr=tmem_address,
    )
    tmem_pool.alloc((128, 160), "float32")
    pool.commit()
    tmem_pool.commit()
    tmem_pool.dealloc()
    if thread_id == 0:
        output[0] = 1


@T.prim_func
def tcgen_dealloc_non_power_of_two_columns():
    T.device_entry()
    T.warp_id([1])
    T.lane_id([32])
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), 160)


_TCGEN_CONTROL_OPS = {
    "tirx.ptx.tcgen05_alloc",
    "tirx.ptx.tcgen05_alloc_exclusive",
    "tirx.ptx.tcgen05_commit",
    "tirx.ptx.tcgen05_commit_multicast",
    "tirx.ptx.tcgen05_commit_multicast_width",
    "tirx.ptx.tcgen05_dealloc",
    "tirx.ptx.tcgen05_dealloc_exclusive",
    "tirx.ptx.tcgen05_fence",
    "tirx.ptx.tcgen05_relinquish_alloc_permit",
    "tirx.ptx.tcgen05_wait",
}


def _site_ops(module: v2.CompiledModule) -> list[str]:
    return [str(site.get("op_name", "")) for site in module.spec.kernels[0].sites]


def _stop(error: v2.ExecutionError) -> dict:
    stops = [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops, error.diagnostics
    return stops[0]


def test_tcgen_control_registry_covers_exact_source_signatures():
    """Port of ``tests/numsim/runtime/test_tcgen_codegen.py::test_tcgen_control_registry_covers_exact_source_signatures``.

    Legacy walked the ``analyze`` source map and decoded each call with
    ``decode_ptx_call``. The copy reads the same source op names from the v2
    module's sites, and the wait/fence actions from the lowered Program
    instructions (``TcgenWait.st``, ``Fence.kind``).
    """

    module = v2.transpile(tcgen_control_calls)
    assert [op for op in _site_ops(module) if op in _TCGEN_CONTROL_OPS] == [
        "tirx.ptx.tcgen05_alloc",
        "tirx.ptx.tcgen05_commit",
        "tirx.ptx.tcgen05_commit_multicast",
        "tirx.ptx.tcgen05_wait",
        "tirx.ptx.tcgen05_wait",
        "tirx.ptx.tcgen05_fence",
        "tirx.ptx.tcgen05_fence",
        "tirx.ptx.tcgen05_relinquish_alloc_permit",
        "tirx.ptx.tcgen05_dealloc",
    ]

    actions = []
    for instr in module.document["kernels"][0]["code"]:
        if not isinstance(instr, dict):
            continue
        if "TcgenWait" in instr:
            actions.append("wait::st" if instr["TcgenWait"]["st"] else "wait::ld")
        elif "Fence" in instr and instr["Fence"]["kind"].startswith("Tcgen05"):
            actions.append(
                {
                    "Tcgen05Before": "fence::before_thread_sync",
                    "Tcgen05After": "fence::after_thread_sync",
                }[instr["Fence"]["kind"]]
            )
    assert actions == [
        "wait::ld",
        "wait::st",
        "fence::before_thread_sync",
        "fence::after_thread_sync",
    ]


def test_tcgen_fence_forms_are_numerical_noops():
    """Port of ``tests/numsim/runtime/test_tcgen_codegen.py::test_tcgen_fence_forms_are_numerical_noops``.

    Dropped pin: ``"tirx.ptx.tcgen05_fence" not in module.rust_source``. The
    ``call_op_names(spec)`` check reads the v2 sites instead (the legacy
    ``source_map`` is empty under v2).
    """

    module = v2.transpile(tcgen_fence_noop_forms)
    result = v2.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 33, dtype=np.uint32))
    assert "tirx.ptx.tcgen05_fence" in _site_ops(module)


def test_frontend_records_dynamic_tmem_lifecycle_from_tcgen_alloc():
    """Port of ``tests/numsim/runtime/test_tcgen_codegen.py::test_frontend_records_dynamic_tmem_lifecycle_from_tcgen_alloc``.

    Legacy ``analyze(...).kernels[0].uses_dynamic_tmem_lifecycle`` becomes the
    v2 Program header flag ``requirements.dynamic_tmem_lifecycle``.
    """

    module = v2.transpile(tcgen_control_calls)
    assert module.document["kernels"][0]["requirements"]["dynamic_tmem_lifecycle"] is True


def test_tmem_pool_non_power_of_two_columns_fail_at_engine():
    """Port of ``tests/numsim/runtime/test_tcgen_codegen.py::test_tmem_pool_non_power_of_two_columns_fail_at_engine``.

    Dropped pin: the legacy ``tcgen_invalid_columns at tcgen05.alloc ... got
    160`` text. Kind: ``sync_protocol_error`` carrying
    ``Tcgen(InvalidColumns { columns: 160 })``.
    """

    module = v2.transpile(tmem_pool_non_power_of_two_columns)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})
    stop = _stop(caught.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "sync_protocol_error", stop
    assert (stop.get("protocol"), stop.get("error"), stop.get("columns")) == ("tcgen", "invalid_columns", 160), stop


def test_tcgen_dealloc_non_power_of_two_columns_fail_at_engine():
    """Port of ``tests/numsim/runtime/test_tcgen_codegen.py::test_tcgen_dealloc_non_power_of_two_columns_fail_at_engine``.

    Dropped pin: the legacy ``tcgen_invalid_columns at tcgen05.dealloc ...
    got 160`` text. Kind: ``sync_protocol_error``. v2 reports the dealloc of
    a never-allocated 160-column range as ``Tcgen(DeallocationMismatch)``
    rather than ``InvalidColumns``; both are the same fail-closed kind, so
    the variant is not pinned.
    """

    module = v2.transpile(tcgen_dealloc_non_power_of_two_columns)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {})
    stop = _stop(caught.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "sync_protocol_error", stop


def test_tcgen_two_cta_lifecycle_rendezvous():
    """Port of ``tests/numsim/runtime/test_tcgen_codegen.py::test_tcgen_two_cta_lifecycle_rendezvous``.

    Dropped pin: ``result.stats["completed_task_count"] == 2``; the run
    status is asserted ``completed`` instead.
    """

    output = np.full(2, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    module = v2.transpile(tcgen_lifecycle_two_cta)
    result = v2.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(2, dtype=np.uint32))
    assert result.status.get("kind") == "completed", result.status


def test_tcgen_dealloc_runtime_rejects_lane_disagreement():
    """Port of ``tests/numsim/runtime/test_tcgen_codegen.py::test_tcgen_dealloc_runtime_rejects_lane_disagreement``.

    Dropped pin: the legacy ``must agree across active lanes`` text. Kind:
    ``divergence`` (the dealloc address operand is not lane-uniform).
    """

    module = v2.transpile(tcgen_runtime_divergent_dealloc)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})
    stop = _stop(caught.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "divergence", stop
