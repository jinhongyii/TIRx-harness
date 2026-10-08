"""v2 ports of ``tests/numsim/integration/test_frontend.py`` (the functions
that used the legacy ``analyze``/``verify``).

Legacy topology facts (``spec.topology.clusters``, ``ctas_per_cluster``,
``warps_per_cta``, ``warp_count``) are read from the v2 module document
(``kernels[0].topology``: ``grid`` in CTAs, ``cluster`` in CTAs per cluster,
``block`` in threads) or from observable outputs. Legacy ``analyze``
rejections become ``v2.transpile`` raising ``UnsupportedTIRxError``; legacy
acceptances become a successful transpile (and run, where the kernel writes
something). Kernels are copied verbatim from the legacy file and from
``tests/numsim/support/kernels.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import NumSimBuildError, UnsupportedTIRxError

pytestmark = requires_v2_engine


def _topology(module) -> dict:
    """``clusters``/``ctas_per_cluster``/``warps_per_cta``/``warp_count`` from
    the v2 document (static launch only)."""

    topology = module.document["kernels"][0]["topology"]
    grid = math.prod(dim["Const"] for dim in topology["grid"])
    cluster = math.prod(topology["cluster"])
    threads = math.prod(topology["block"])
    assert threads % 32 == 0, topology
    return {
        "clusters": grid // cluster,
        "ctas_per_cluster": cluster,
        "warps_per_cta": threads // 32,
        "warp_count": grid * threads // 32,
    }


def _unsupported(module) -> list:
    return list(module.document["kernels"][0]["unsupported"])


# -- kernels copied from tests/numsim/support/kernels.py ----------------------


@T.prim_func
def no_op_kernel():
    T.device_entry()
    _cta = T.cta_id([2])
    _warp = T.warp_id([3])
    _lane = T.lane_id([32])


@T.prim_func
def warpgroup_scope_coordinates(output: T.Buffer((2, 4, 32, 3), "int32")):
    T.device_entry()
    _cta = T.cta_id([1])
    warpgroup = T.warpgroup_id([2])
    warp_in_group = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread_in_group = T.thread_id_in_wg([128])
    output[warpgroup, warp_in_group, lane, 0] = warpgroup
    output[warpgroup, warp_in_group, lane, 1] = warp_in_group
    output[warpgroup, warp_in_group, lane, 2] = thread_in_group


@T.prim_func
def unsupported_exp(source: T.Buffer((32,), "float32"), destination: T.Buffer((32,), "float32")):
    T.device_entry()
    lane = T.lane_id([32])
    destination[lane] = T.sin(source[lane])


@T.prim_func
def lane_add(
    left: T.Buffer((100,), "float32"),
    right: T.Buffer((100,), "float32"),
    output: T.Buffer((100,), "float32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    index = T.meta_var(warp * 32 + lane)
    if index < 100:
        output[index] = left[index] + right[index]


@T.prim_func
def bound_launch_topology():
    T.device_entry()
    max_ctas: T.let = T.int32(4)
    should_cap: T.let = max_ctas > T.int32(3)
    cta_count: T.let = T.Select(should_cap, T.min(max_ctas - T.int32(1), T.int32(3)), T.int32(1))
    _cta = T.cta_id([cta_count])
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])


# -- kernels copied from tests/numsim/integration/test_frontend.py -----------


@T.prim_func(check_well_formed=False)
def conflicting_repeated_warp_extents():
    T.device_entry()
    _warp_a = T.warp_id([2])
    _warp_b = T.warp_id([3])
    _lane = T.lane_id([32])


@T.prim_func(check_well_formed=False)
def invalid_warpgroup_warp_extent():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([3])
    _lane = T.lane_id([32])


@T.prim_func(check_well_formed=False)
def invalid_warpgroup_thread_extent():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _thread = T.thread_id_in_wg([96])


@T.prim_func(check_well_formed=False)
def conflicting_direct_and_warpgroup_extents():
    T.device_entry()
    _warpgroup = T.warpgroup_id([2])
    _warp = T.warp_id([4])
    _lane = T.lane_id([32])


@T.prim_func(check_well_formed=False)
def direct_warps_with_warpgroup_local_thread_coordinate():
    T.device_entry()
    _warp = T.warp_id([8])
    _thread = T.thread_id_in_wg([128])


@T.prim_func(check_well_formed=False)
def conflicting_cluster_extents():
    T.device_entry()
    _cluster = T.cluster_id([2])
    _global_cta = T.cta_id([6])
    _cluster_cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])


@T.prim_func
def multidimensional_cluster_with_cta_pair():
    T.device_entry()
    _cta_x, _cta_y = T.cta_id_in_cluster([4, 2])
    _pair = T.cta_id_in_pair()
    _lane = T.lane_id([32])


@T.prim_func
def launch_metadata_attr(output: T.Buffer((1,), "int32")):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = 1


@T.prim_func
def unknown_attr(output: T.Buffer((1,), "int32")):
    T.device_entry()
    T.attr({"numsim.unknown_control": 1})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = 1


@T.prim_func
def supported_loop_metadata(output: T.Buffer((8,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.serial(2, unroll=False):
        output[index] = index
    for index in T.unroll(2, 4):
        output[index] = index
    for index in T.serial(4, 6, unroll=True):
        output[index] = index
    for index in T.serial(6, 8, unroll=2):
        output[index] = index


@T.prim_func
def unknown_loop_annotation(output: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.serial(4, annotations={"numsim.unknown_loop": 1}):
        output[index] = index


@T.prim_func
def vectorized_loop(output: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.vectorized(4):
        output[index] = index


@T.prim_func
def parallel_loop(output: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.parallel(4):
        output[index] = index


@T.prim_func
def thread_bound_loop(output: T.Buffer((4,), "int32")):
    T.device_entry()
    for index in T.thread_binding(4, thread="threadIdx.x"):
        output[index] = index


# -- tests --------------------------------------------------------------------


def test_source_spans_exclude_process_local_source_name_addresses():
    """Port of ``tests/numsim/integration/test_frontend.py::test_source_spans_exclude_process_local_source_name_addresses``.

    The spans come from the v2 ``KernelSpec.sites[*].spans`` instead of the
    legacy ``source_map``. ``lane_add`` is copied into this file, so the span
    file is this file (legacy: ``kernels.py``, where it was defined)."""

    module = v2.transpile(lane_add)
    spans = [
        json.dumps(span, sort_keys=True)
        for site in module.spec.kernels[0].sites
        for span in site.get("spans", ())
    ]

    assert spans
    assert all("0x" not in span for span in spans)
    assert all(Path(__file__).name in span for span in spans)


def test_no_op_topology_is_one_future_per_warp():
    """Port of ``tests/numsim/integration/test_frontend.py::test_no_op_topology_is_one_future_per_warp``.

    Topology from the v2 document; ``verify(spec)`` is a successful transpile
    and run. Dropped: the legacy "future per warp" notion itself (only the
    warp count is observable)."""

    module = v2.transpile(no_op_kernel)
    assert _topology(module) == {
        "clusters": 2,
        "ctas_per_cluster": 1,
        "warps_per_cta": 3,
        "warp_count": 6,
    }
    assert _unsupported(module) == []
    result = v2.Engine().run(module, {})
    assert result.status["kind"] == "completed", result.status


def test_warpgroup_scope_topology_uses_four_warps_per_group():
    """Port of ``tests/numsim/integration/test_frontend.py::test_warpgroup_scope_topology_uses_four_warps_per_group``.

    ``warps_per_warpgroup == 4`` and ``threads_per_warpgroup == 128`` are not
    in the v2 document; they are asserted through the coordinates the kernel
    writes (warp-in-group 0..3, thread-in-group 0..127 per group)."""

    module = v2.transpile(warpgroup_scope_coordinates)
    topology = _topology(module)
    assert topology["clusters"] == 1
    assert topology["ctas_per_cluster"] == 1
    assert topology["warps_per_cta"] == 8
    assert _unsupported(module) == []

    result = v2.Engine().run(module, {"output": np.zeros((2, 4, 32, 3), dtype=np.int32)})
    group, warp, lane = np.meshgrid(np.arange(2), np.arange(4), np.arange(32), indexing="ij")
    expected = np.stack([group, warp, warp * 32 + lane], axis=-1).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_warpgroup_local_coordinate_does_not_imply_one_group_per_cta():
    """Port of ``tests/numsim/integration/test_frontend.py::test_warpgroup_local_coordinate_does_not_imply_one_group_per_cta``.

    Dropped: ``warps_per_warpgroup == 4`` (not in the v2 document and the
    kernel writes nothing)."""

    module = v2.transpile(direct_warps_with_warpgroup_local_thread_coordinate)
    assert _topology(module)["warps_per_cta"] == 8


@v2_gap("bound_launch_topology: document grid is 1 CTA; the let-bound extent Select(4 > 3, min(3, 3), 1) is 3")
def test_bound_launch_extent_is_resolved_in_statement_order():
    """Port of ``tests/numsim/integration/test_frontend.py::test_bound_launch_extent_is_resolved_in_statement_order``."""

    module = v2.transpile(bound_launch_topology)
    assert _unsupported(module) == []
    topology = _topology(module)
    assert topology["clusters"] == 3
    assert topology["ctas_per_cluster"] == 1
    assert topology["warps_per_cta"] == 2


@T.prim_func
def if_then_else_launch_extent(choose: T.bool, count: T.int32):
    T.device_entry()
    _warp = T.warp_id([T.if_then_else(choose, count, 1)])
    _lane = T.lane_id([32])


def _specialize(**bindings):
    choose, count = if_then_else_launch_extent.params
    params = {"choose": choose, "count": count}
    return if_then_else_launch_extent.specialize({params[k]: v for k, v in bindings.items()})


def test_constant_if_then_else_launch_extent_only_evaluates_selected_branch():
    """Port of ``tests/numsim/integration/test_frontend.py::test_constant_if_then_else_launch_extent_only_evaluates_selected_branch`` (``choose=False`` half).

    The ``choose=True, count=2`` half and the fail-closed half are split into
    the two tests below (both ``v2_gap``)."""

    assert _topology(v2.transpile(_specialize(choose=False)))["warps_per_cta"] == 1


@v2_gap("if_then_else(True, 2, 1) warp extent: document block is 32 threads (1 warp), expected 2 warps")
def test_constant_if_then_else_launch_extent_only_evaluates_selected_branch_true():
    """Port of ``tests/numsim/integration/test_frontend.py::test_constant_if_then_else_launch_extent_only_evaluates_selected_branch`` (``choose=True, count=2`` half)."""

    assert _topology(v2.transpile(_specialize(choose=True, count=2)))["warps_per_cta"] == 2


@pytest.mark.parametrize(
    "bindings",
    [
        pytest.param({"choose": True}, id="count-unbound"),
        pytest.param({"count": 2}, id="choose-unbound"),
    ],
)
@v2_gap("launch extent that is not statically known is accepted (document block 32 threads), expected UnsupportedTIRxError")
def test_constant_if_then_else_launch_extent_not_static_fails_closed(bindings):
    """Port of ``tests/numsim/integration/test_frontend.py::test_constant_if_then_else_launch_extent_only_evaluates_selected_branch`` (fail-closed half).

    Dropped: the legacy message "launch extent is not statically known"."""

    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(_specialize(**bindings))


def test_cta_pair_rank_does_not_override_cluster_cta_extent():
    """Port of ``tests/numsim/integration/test_frontend.py::test_cta_pair_rank_does_not_override_cluster_cta_extent``."""

    module = v2.transpile(multidimensional_cluster_with_cta_pair)
    assert _unsupported(module) == []
    topology = _topology(module)
    assert topology["clusters"] == 1
    assert topology["ctas_per_cluster"] == 8
    assert topology["warps_per_cta"] == 1


@pytest.mark.parametrize(
    "kernel",
    [
        conflicting_repeated_warp_extents,
        invalid_warpgroup_warp_extent,
        invalid_warpgroup_thread_extent,
        conflicting_direct_and_warpgroup_extents,
        conflicting_cluster_extents,
    ],
    ids=[
        "conflicting_repeated_warp_extents",
        "invalid_warpgroup_warp_extent",
        "invalid_warpgroup_thread_extent",
        "conflicting_direct_and_warpgroup_extents",
        "conflicting_cluster_extents",
    ],
)
@v2_gap("conflicting/invalid launch-topology constraints are accepted (transpile and run succeed), expected a fail-closed rejection")
def test_conflicting_or_invalid_topology_constraints_fail_closed(kernel):
    """Port of ``tests/numsim/integration/test_frontend.py::test_conflicting_or_invalid_topology_constraints_fail_closed``.

    Dropped: the legacy messages. The rejection may come from ``transpile``
    or from launching the module, but it must come."""

    with pytest.raises((UnsupportedTIRxError, NumSimBuildError)):
        v2.Engine().run(v2.transpile(kernel), {})


def test_numerically_irrelevant_launch_metadata_is_explicitly_supported():
    """Port of ``tests/numsim/integration/test_frontend.py::test_numerically_irrelevant_launch_metadata_is_explicitly_supported``."""

    module = v2.transpile(launch_metadata_attr)
    assert _unsupported(module) == []
    result = v2.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], [1])


def test_unknown_attr_semantics_fail_closed():
    """Port of ``tests/numsim/integration/test_frontend.py::test_unknown_attr_semantics_fail_closed``.

    Dropped: the legacy message; the ``unsupported`` entry still names the
    attribute key."""

    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(unknown_attr)
    assert any("unknown_control" in entry for entry in excinfo.value.unsupported), excinfo.value.unsupported


def test_serial_and_unrolled_loop_metadata_are_explicitly_supported():
    """Port of ``tests/numsim/integration/test_frontend.py::test_serial_and_unrolled_loop_metadata_are_explicitly_supported``."""

    module = v2.transpile(supported_loop_metadata)
    assert _unsupported(module) == []
    result = v2.Engine().run(module, {"output": np.full(8, -1, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], np.arange(8, dtype=np.int32))


@v2_gap("for-loop with an unknown annotation 'numsim.unknown_loop' is accepted, expected UnsupportedTIRxError naming it")
def test_unknown_loop_annotations_fail_closed():
    """Port of ``tests/numsim/integration/test_frontend.py::test_unknown_loop_annotations_fail_closed``."""

    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(unknown_loop_annotation)
    assert any("unknown_loop" in entry for entry in excinfo.value.unsupported), excinfo.value.unsupported


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(
            vectorized_loop,
            marks=v2_gap("T.vectorized loop is accepted and run sequentially, expected UnsupportedTIRxError"),
            id="vectorized_loop-VECTORIZED",
        ),
        pytest.param(parallel_loop, id="parallel_loop-PARALLEL"),
        pytest.param(thread_bound_loop, id="thread_bound_loop-THREAD_BINDING"),
    ],
)
def test_non_serial_loop_semantics_are_not_silently_sequentialized(kernel):
    """Port of ``tests/numsim/integration/test_frontend.py::test_non_serial_loop_semantics_are_not_silently_sequentialized``.

    Dropped: the legacy ``ForKind`` name in the message (v2 names the
    ``tirx.For`` node and the numeric loop kind)."""

    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(kernel)
    assert any("tirx.For" in entry for entry in excinfo.value.unsupported), excinfo.value.unsupported


def test_unknown_reachable_nodes_fail_closed():
    """Port of ``tests/numsim/integration/test_frontend.py::test_unknown_reachable_nodes_fail_closed``.

    Delta D3 (numsim-behaviour-deltas.md): ``T.sin`` was unknown to legacy and
    rejected; v2 implements it with host libm. The kernel now transpiles and
    computes ``sin`` (f32, host-libm value)."""

    module = v2.transpile(unsupported_exp)
    assert _unsupported(module) == []
    source = np.linspace(-3, 3, 32, dtype=np.float32)
    result = v2.Engine().run(module, {"source": source, "destination": np.zeros(32, dtype=np.float32)})
    np.testing.assert_allclose(result.outputs["destination"], np.sin(source), rtol=1e-6, atol=1e-7)
