"""v2 copies of legacy fail-closed tests that pinned legacy error text.

``test-migration.md`` ("Public-API A tests under ``NUMSIM_IMPL=v2``") rules
these A: v2 raises the same exception type for the same fault, only the
wording differs. Each copy asserts the exception type and the v2 error kind
(plus the structured detail where it is part of the kind, e.g.
``MissingWarpgroupSync``), never the human text. Kernels are copied verbatim
from the legacy files.
"""

from __future__ import annotations

from collections.abc import Iterable

import ml_dtypes
import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import NumSimExecutionError, UnsupportedTIRxError
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap

pytestmark = requires_v2_engine


def _first_stop(error: v2.ExecutionError) -> dict:
    """The diagnostic ``Engine.run`` raised for (same selection as run.py)."""

    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


def _anchor_text(stop: dict, source: str | None) -> str:
    """Text of the source line the stopping diagnostic is anchored at."""
    span = stop.get("source_span") or {}
    assert span.get("kind") == "span", stop
    line = span["line"]
    if source is None:
        with open(span["source_name"]) as handle:
            lines = handle.read().splitlines()
    else:
        lines = source.splitlines()
    return lines[line - 1]


def _assert_stop(
    excinfo,
    kinds: Iterable[str],
    detail: str | None = None,
    *,
    anchor: str | None = None,
    source: str | None = None,
    lanes: int | None = None,
    warp: int | None = None,
) -> None:
    """``ExecutionError`` whose stopping diagnostic is an ``error`` of one of
    ``kinds``; ``detail`` is the structured variant name the kind carries
    (e.g. ``RegPool(MissingWarpgroupSync {..})``).

    W11 (pin-message): the legacy text also carried *where* the fault is, so
    ``anchor`` checks the source line of the diagnostic's ``source_span``
    (``source`` is the TVMScript text for ``from_source`` kernels), and
    ``lanes`` / ``warp`` check the structured faulting lanes and warp."""

    error = excinfo.value
    assert isinstance(error, v2.ExecutionError), type(error)
    stop = _first_stop(error)
    assert stop["status"] == "error", stop
    assert stop["kind"] in set(kinds), stop
    if detail is not None:
        assert detail in str(stop.get("message", "")), stop
    if anchor is not None:
        assert anchor in _anchor_text(stop, source), (anchor, stop)
    if lanes is not None:
        assert stop.get("lanes") == f"WarpMask(0x{lanes:08x})", stop
    if warp is not None:
        assert stop.get("warp") == warp, stop


def _run(kernel, inputs, **kwargs):
    return v2.Engine().run(v2.transpile(kernel), inputs, **kwargs)


# -- tests/numsim/integration/test_host_prelude.py ---------------------------


@T.prim_func
def host_encoded_unregistered_integer_tensor_map(source: T.Buffer((32,), "uint8"), delta: T.int32):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "uint8",
        1,
        source.data,
        T.bitwise_and(delta, T.int32(31)) + T.int32(1),
        16,
        1,
        0,
        0,
        0,
        0,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])


def test_host_tensor_map_integer_expressions_fail_closed_on_unregistered_nodes():
    """Replaces ``tests/numsim/integration/test_host_prelude.py::test_host_tensor_map_integer_expressions_fail_closed_on_unregistered_nodes``.

    Kind: transpile-time ``UnsupportedTIRxError`` whose ``unsupported`` entry
    names the ``BitwiseAnd`` node."""

    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(host_encoded_unregistered_integer_tensor_map)
    assert any("BitwiseAnd" in entry for entry in excinfo.value.unsupported), excinfo.value.unsupported


# -- tests/numsim/integration/test_warp_ops_artifact.py ----------------------


@T.prim_func
def warp_participant_contract(
    participant_masks: T.Buffer((32,), "uint32"),
    active_count: T.Buffer((1,), "int32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < active_count[0]:
        output[lane] = T.cuda.ballot_sync(participant_masks[lane], lane < 16)


def test_warp_collectives_reject_invalid_participant_contracts():
    """Replaces ``tests/numsim/integration/test_warp_ops_artifact.py::test_warp_collectives_reject_invalid_participant_contracts``.

    Every invalid membermask (zero, missing the executing lane, lane-varying,
    naming an inactive lane) is the ``divergence`` kind."""

    module = v2.transpile(warp_participant_contract)
    active_count = np.array([32], dtype=np.int32)
    output = np.zeros(32, dtype=np.uint32)

    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {
                "participant_masks": np.zeros(32, dtype=np.uint32),
                "active_count": active_count,
                "output": output,
            },
        )
    _assert_stop(excinfo, {"divergence", "warp_collective_divergence"}, anchor="T.cuda.ballot_sync", lanes=0xFFFFFFFF)

    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {
                "participant_masks": np.full(32, np.uint32(0x7FFFFFFF), dtype=np.uint32),
                "active_count": active_count,
                "output": output,
            },
        )
    _assert_stop(excinfo, {"divergence", "warp_collective_divergence"}, anchor="T.cuda.ballot_sync", lanes=0xFFFFFFFF)

    inconsistent = np.full(32, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    inconsistent[7] = np.uint32(0xFFFFFFFE)
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {"participant_masks": inconsistent, "active_count": active_count, "output": output},
        )
    _assert_stop(excinfo, {"divergence", "warp_collective_divergence"}, anchor="T.cuda.ballot_sync", lanes=0xFFFFFFFF)

    active_count[0] = np.int32(16)
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {
                "participant_masks": np.full(32, np.uint32(0xFFFFFFFF), dtype=np.uint32),
                "active_count": active_count,
                "output": output,
            },
        )
    _assert_stop(excinfo, {"divergence", "warp_collective_divergence"}, anchor="T.cuda.ballot_sync", lanes=0x0000FFFF)


# -- tests/numsim/runtime/test_dynamic_pure_call_runtime_domains.py ----------


@T.prim_func
def IF_THEN_ELSE_MIXED_POINTER_SPACES(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    shared[lane] = T.uint32(0xA7)
    T.cuda.warp_sync()
    selected: T.let[T.handle] = T.call_intrin(
        "handle",
        "prim.if_then_else",
        lane % 2 == 0,
        source.ptr_to([lane]),
        shared.ptr_to([lane]),
    )
    T.ptx.ld.global_.u32(output[lane], selected)


def test_if_then_else_mixed_pointer_spaces_fail_closed():
    """Replaces ``tests/numsim/runtime/test_dynamic_pure_call_runtime_domains.py::test_if_then_else_mixed_pointer_spaces_fail_closed``.

    A shared pointer used by ``ld.global`` is the ``bad_address`` kind."""

    module = v2.transpile(IF_THEN_ELSE_MIXED_POINTER_SPACES)
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine(max_workers=1).run(
            module,
            {
                "source": np.full(32, 0x35, dtype=np.uint32),
                "output": np.zeros(32, dtype=np.uint32),
            },
        )
    _assert_stop(excinfo, {"bad_address"}, anchor="T.ptx.ld.global_", lanes=0x2)


# -- tests/numsim/runtime/test_memory_coverage_next.py -----------------------


@pytest.mark.parametrize("address", ["source.ptr_to([lane])", "source.ptr_to([1])"])
def test_ldu_rejects_nonuniform_or_misaligned_vector(address):
    """Replaces ``tests/numsim/runtime/test_memory_coverage_next.py::test_ldu_rejects_nonuniform_or_misaligned_vector``.

    Legacy accepted either message ("lane-varying" or the 8-byte alignment
    rule) for either param; v2 kinds ``divergence`` or ``misaligned``."""

    kernel_source = f"""
@T.prim_func
def invalid(source: T.Buffer((64,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    out = T.alloc_local((2,), "uint32")
    T.ptx.ldu.global_.v2.u32(out[0], out[1], {address})
"""
    kernel = tvm.script.from_source(kernel_source, {"T": T})
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(v2.transpile(kernel), {"source": np.zeros(64, np.uint32)})
    # The first faulting lane: lane 1 (byte offset 4) for ptr_to([lane]), lane 0 for ptr_to([1]).
    _assert_stop(
        excinfo,
        {"divergence", "warp_collective_divergence", "misaligned"},
        anchor="T.ptx.ldu.global_",
        source=kernel_source,
        lanes=0x2 if "lane" in address else 0x1,
    )


# -- tests/numsim/runtime/test_ordering_calls.py -----------------------------


@T.prim_func
def setmaxnreg_without_intervening_sync(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(256)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_warp_disagreement(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    if warp == 0:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    else:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(32)
    if (warp == 0) and (lane == 0):
        output[0] = 7


def test_setmaxnreg_requires_explicit_warpgroup_sync_before_a_later_call():
    """Replaces ``tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_requires_explicit_warpgroup_sync_before_a_later_call``."""

    module = v2.transpile(setmaxnreg_without_intervening_sync)
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})
    _assert_stop(excinfo, {"sync_protocol_error"}, detail="MissingWarpgroupSync", anchor="setmaxnreg.dec", warp=0)


def test_setmaxnreg_rejects_warp_disagreement_within_one_occurrence():
    """Replaces ``tests/numsim/runtime/test_ordering_calls.py::test_setmaxnreg_rejects_warp_disagreement_within_one_occurrence``."""

    module = v2.transpile(setmaxnreg_warp_disagreement)
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})
    _assert_stop(excinfo, {"divergence", "warp_collective_divergence"}, anchor="setmaxnreg.inc", warp=1)


# -- tests/numsim/runtime/test_packed_float4_global_views.py -----------------


@T.prim_func
def read_four_float4(source: T.Buffer((4,), "float4_e2m1fn"), output: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        output[lane] = T.cast(source[lane], "float32")


@pytest.fixture(scope="module")
def four_float4_module():
    return v2.transpile(read_four_float4)


def test_direct_one_byte_per_value_float4_array_is_rejected(four_float4_module):
    """Replaces ``tests/numsim/runtime/test_packed_float4_global_views.py::test_direct_one_byte_per_value_float4_array_is_rejected``.

    Kind: the input binding rejects the array (``v2.InputError``, a
    ``NumSimExecutionError``) before anything runs."""

    unpacked = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.float4_e2m1fn)

    with pytest.raises(v2.InputError):
        v2.Engine().run(
            four_float4_module,
            {"source": unpacked, "output": np.zeros(4, dtype=np.float32)},
        )


# -- tests/numsim/runtime/test_ptx_integer_arithmetic.py ---------------------


@T.prim_func
def ptx_div_s32_error_path(
    lhs: T.Buffer((32,), "int32"),
    rhs: T.Buffer((32,), "int32"),
    output: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.div.s32(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def ptx_rem_s32_error_path(
    lhs: T.Buffer((32,), "int32"),
    rhs: T.Buffer((32,), "int32"),
    output: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.rem.s32(output[lane], lhs[lane], rhs[lane])


@pytest.mark.parametrize(
    ("kernel", "symbol"),
    ((ptx_div_s32_error_path, "/"), (ptx_rem_s32_error_path, "%")),
    ids=("div", "rem"),
)
def test_ptx_integer_division_by_zero_fails_closed_at_the_faulting_lane(kernel, symbol):
    """Replaces ``tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_integer_division_by_zero_fails_closed_at_the_faulting_lane``.

    Kind ``invalid_operand`` (legacy also named lane 7 in its text; v2's
    diagnostic reports the warp, not the lane)."""

    rhs = np.ones(32, dtype=np.int32)
    rhs[7] = 0
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            v2.transpile(kernel),
            {
                "lhs": np.full(32, 29, dtype=np.int32),
                "rhs": rhs,
                "output": np.zeros(32, dtype=np.int32),
            },
            outputs=("output",),
        )
    _assert_stop(excinfo, {"invalid_operand"}, anchor="T.ptx.div" if symbol == "/" else "T.ptx.rem")


@pytest.mark.parametrize(
    ("kernel", "symbol"),
    ((ptx_div_s32_error_path, "/"), (ptx_rem_s32_error_path, "%")),
    ids=("div", "rem"),
)
def test_ptx_signed_division_overflow_fails_closed(kernel, symbol):
    """Replaces ``tests/numsim/runtime/test_ptx_integer_arithmetic.py::test_ptx_signed_division_overflow_fails_closed``."""

    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            v2.transpile(kernel),
            {
                "lhs": np.full(32, np.iinfo(np.int32).min, dtype=np.int32),
                "rhs": np.full(32, -1, dtype=np.int32),
                "output": np.zeros(32, dtype=np.int32),
            },
            outputs=("output",),
        )
    _assert_stop(excinfo, {"invalid_operand"}, anchor="T.ptx.div" if symbol == "/" else "T.ptx.rem")


# -- tests/numsim/runtime/test_scalar_control.py -----------------------------


@T.prim_func
def mbarrier_stale_state_token(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    tokens = T.alloc_buffer((3,), "uint64", scope="local")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), T.uint32(1))
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        barrier_address: T.let = T.cuda.cvta_generic_to_shared(T.address_of(barrier[0]))
        for generation in T.serial(3):
            T.ptx.mbarrier.arrive.shared__cta.b64(tokens[generation], barrier_address, T.uint32(1))
            T.ptx.mbarrier.try_wait.shared__cta.b64(
                output[generation], barrier_address, tokens[generation]
            )
        T.ptx.mbarrier.try_wait.shared__cta.b64(output[3], barrier_address, tokens[0])


@T.prim_func
def integer_trap_predicate(flag: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.cuda.trap_when_assert_failed(flag[0])
    output[lane] = lane + 1


@T.prim_func
def floating_trap_predicate(flag: T.Buffer((1,), "float32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.cuda.trap_when_assert_failed(flag[0])
    output[lane] = lane + 1


@T.prim_func
def misaligned_cuda_pointer_helpers(
    source: T.Buffer((32,), "uint8"), destination: T.Buffer((32,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.cuda.float22half2(T.address_of(destination[1]), T.address_of(source[1]))


def test_mbarrier_state_token_rejects_a_generation_older_than_the_previous_one():
    """Replaces ``tests/numsim/runtime/test_scalar_control.py::test_mbarrier_state_token_rejects_a_generation_older_than_the_previous_one``."""

    module = v2.transpile(mbarrier_stale_state_token)
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(4, dtype=np.uint32)})
    _assert_stop(excinfo, {"sync_protocol_error"}, detail="InvalidStateToken", anchor="mbarrier.try_wait", lanes=0x1)


def test_integer_trap_predicate_uses_cpp_truth_conversion():
    """Replaces ``tests/numsim/runtime/test_scalar_control.py::test_integer_trap_predicate_uses_cpp_truth_conversion``."""

    module = v2.transpile(integer_trap_predicate)
    passed = v2.Engine().run(
        module, {"flag": np.array([7], dtype=np.int32), "output": np.zeros(32, dtype=np.int32)}
    )
    np.testing.assert_array_equal(passed.outputs["output"], np.arange(1, 33, dtype=np.int32))

    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            module, {"flag": np.array([0], dtype=np.int32), "output": np.zeros(32, dtype=np.int32)}
        )
    _assert_stop(excinfo, {"trap"}, anchor="trap_when_assert_failed")


@pytest.mark.parametrize("predicate", [np.float32(0.0), np.float32(-0.0)])
def test_floating_trap_rejects_signed_zero(predicate):
    """Replaces ``tests/numsim/runtime/test_scalar_control.py::test_floating_trap_rejects_signed_zero``."""

    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            v2.transpile(floating_trap_predicate),
            {
                "flag": np.array([predicate], dtype=np.float32),
                "output": np.zeros(32, dtype=np.int32),
            },
        )
    _assert_stop(excinfo, {"trap"}, anchor="trap_when_assert_failed")


def test_cuda_pointer_helpers_check_typed_dereference_alignment():
    """Replaces ``tests/numsim/runtime/test_scalar_control.py::test_cuda_pointer_helpers_check_typed_dereference_alignment``."""

    module = v2.transpile(misaligned_cuda_pointer_helpers)
    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(
            module,
            {"source": np.arange(32, dtype=np.uint8), "destination": np.zeros(32, dtype=np.uint8)},
        )
    _assert_stop(excinfo, {"misaligned"}, anchor="float22half2", lanes=0x1)


# -- tests/numsim/runtime/test_wait_until.py ---------------------------------
# Kernel of ``tests/numsim/support/wait_until.py::indexed_predicate_case``
# with ``alias=False, table_size=2`` (that module imports
# ``tirx_harness.numsim.cases``, so the builder is copied, concretized).

_INDEXED_PREDICATE = """
@T.prim_func
def indexed(state: T.Buffer((32,), "int32"), out: T.Buffer((32,), "int32")):
    T.device_entry()
    T.cta_id([1])
    lane = T.thread_id([32])
    seen = T.alloc_local((2,), "int32")
    table = T.alloc_local((2,), "int32")
    seen[1] = 1
    table[0] = 0
    table[1] = 1
    T.cuda.wait_until(seen[0], state.ptr_to([lane]), lambda current: table[current] == 1)
    out[lane] = seen[0]
"""


def test_a_candidate_index_outside_its_local_table_is_an_execution_error():
    """Replaces ``tests/numsim/runtime/test_wait_until.py::test_a_candidate_index_outside_its_local_table_is_an_execution_error``."""

    kernel = tvm.script.from_source(_INDEXED_PREDICATE, {"T": T})
    inputs = {"state": np.full(32, 2, np.int32), "out": np.zeros(32, np.int32)}
    module = v2.transpile(kernel)

    with pytest.raises(NumSimExecutionError) as excinfo:
        v2.Engine().run(module, inputs)
    _assert_stop(excinfo, {"out_of_bounds"}, anchor="wait_until", source=_INDEXED_PREDICATE, lanes=0x1)
