"""v2 ports of ``tests/numsim/integration/test_artifact_build.py`` (step-5 list B1).

Kernels are copied verbatim from ``tests/numsim/support/kernels.py`` and
``tests/numsim/support/remote_mbarrier.py``. Dropped everywhere: the legacy
``module.rust_source`` text pins, ``result.stats`` scheduler counters
(``task_count``, ``completed_task_count``, ``poll_order``, ``worker_count``,
``scheduling_domain_count``, completion-pump counters) and ``cache_dir=tmp_path``.
Error tests assert the exception type and the v2 diagnostic kind, never text.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from tvm import tirx
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm_ffi import structural_equal, structural_hash

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import NumSimExecutionError

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


def _assert_stop(excinfo, kinds: Iterable[str]) -> dict:
    """``ExecutionError`` whose stopping diagnostic is an ``error`` of one of ``kinds``."""

    error = excinfo.value
    assert isinstance(error, v2.ExecutionError), type(error)
    stop = _first_stop(error)
    assert stop["status"] == "error", stop
    assert stop["kind"] in set(kinds), stop
    return stop


def _run(kernel, inputs, **kwargs):
    engine_kwargs = {k: kwargs.pop(k) for k in ("max_workers",) if k in kwargs}
    result = v2.Engine(**engine_kwargs).run(v2.transpile(kernel), inputs, **kwargs)
    assert result.status.get("kind") == "completed", result.status
    return result


def _encode_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _mapa_u64(ptr, rank):
    mapped = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


# -- kernels (verbatim from tests/numsim/support/kernels.py) -----------------


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
def elect_sync_integer_branch(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if T.cuda.elect_sync():
        output[lane] = 1
    else:
        output[lane] = 2


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
def matrix_add_2d(
    left: T.Buffer((3, 5), "float32"),
    right: T.Buffer((3, 5), "float32"),
    output: T.Buffer((3, 5), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 15:
        row = T.meta_var(lane // 5)
        col = T.meta_var(lane % 5)
        output[row, col] = left[row, col] + right[row, col]


@T.prim_func
def shared_alias_per_cta(output: T.Buffer((2, 32), "float32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "float32", scope="shared")
    alias = T.decl_buffer((32,), "float32", data=shared.data, elem_offset=16, scope="shared")
    alias[lane] = T.cast(cta * 100 + lane, "float32")
    output[cta, lane] = shared[16 + lane]


@T.prim_func
def local_array_per_lane(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_buffer((2,), "float32", scope="local")
    local[0] = T.cast(lane, "float32")
    local[1] = T.cast(lane + 1, "float32")
    output[lane] = local[0] + local[1]


@T.prim_func
def divergent_loop_control(output: T.Buffer((128,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in T.serial(4):
        if lane == step:
            continue
        if lane < 4:
            if step == 2:
                break
        output[lane * 4 + step] = T.float32(1)


@T.prim_func
def divergent_while_control(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    while lane < 16:
        output[lane] = T.float32(2)
        break


@T.prim_func
def mbarrier_wait_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        output[lane] = 1


@T.prim_func
def bar_sync_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.ptx.bar.sync(T.uint32(lane), T.uint32(32))
        output[lane] = 1


@T.prim_func
def cta_sync_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.cuda.cta_sync()
        output[lane] = 1


@T.prim_func
def dynamic_for_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for _step in T.serial(1):
        if lane >= 0:
            continue
        for _inner in T.serial(0, lane):
            output[lane] = 1


@T.prim_func
def scalar_expression_mix(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    wide_lane: T.let = T.cast(lane, "int64")
    shifted: T.let = wide_lane - T.int64(17)
    quotient: T.let = shifted // T.int64(5)
    remainder: T.let = shifted % T.int64(5)
    clamped: T.let = T.min(T.max(quotient, T.int64(-2)), T.int64(2))
    use_clamped: T.let = ((lane < 5) and not (lane == 2)) or lane >= 29
    selected: T.let = T.Select(use_clamped, clamped, remainder)
    output[lane] = T.cast(selected, "float32")


@T.prim_func
def nested_mask_parent_scope(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    enabled: T.let = T.bool(True)
    disabled: T.let = T.bool(False)
    if lane < 16:
        output[lane] = 1
        if (lane < 4) or enabled:
            output[lane] = output[lane] + 1
        if not (lane < 8):
            output[lane] = output[lane] + 2
        if disabled and lane < 8:
            output[lane] = output[lane] + 4


@T.prim_func
def guarded_if_then_else_load(
    source: T.Buffer((1,), "float32"), output: T.Buffer((32,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.if_then_else(lane == 0, source[lane], T.float32(7))


@T.prim_func
def dynamic_rows(input_ptr: T.handle, output_ptr: T.handle):
    rows = T.int32()
    input_buffer = T.match_buffer(input_ptr, (rows, 32), "float32")
    output_buffer = T.match_buffer(output_ptr, (rows, 32), "float32")
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    row: T.int32 = 0
    while row < rows:
        output_buffer[row, lane] = input_buffer[row, lane] + T.float32(2)
        row = row + 1


@T.prim_func
def mbarrier_phase_reuse(source: T.Buffer((4,), "float32"), output: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    shared = T.alloc_buffer((4,), "float32", scope="shared", align=128)
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    for phase in T.serial(2):
        if warp == 0:
            if lane == 0:
                if phase == 0:
                    T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
                else:
                    Tx.copy_async(
                        shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0])
                    )
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        else:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(barriers[0]), phase)
                output[phase] = phase + 1
        T.cuda.cta_sync()


@T.prim_func
def mbarrier_missing_arrivals():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 64)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def scoped_syncs(output: T.Buffer((2, 4, 32), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.cuda.warp_sync()
    T.cuda.warpgroup_sync(7)
    T.ptx.bar.sync(T.uint32(6), T.uint32(128))
    T.cuda.cta_sync()
    T.cuda.cluster_sync()
    output[cta, warp, lane] = cta * 10000 + warp * 100 + lane


@T.prim_func
def remote_shared_write_ownership(output: T.Buffer((2,), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float32", scope="shared")
    if lane == 0:
        shared[0] = T.float32(-1)
    T.cuda.cluster_sync()
    if cta == 0:
        if lane < 2:
            mapped = T.alloc_local((1,), "uint64")
            T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(lane))
            remote_ptr: T.let[
                T.Var(name="remote_write_ptr", ty=PointerType(PrimType("float32"), "shared"))
            ] = T.reinterpret(
                PointerType(PrimType("float32"), "shared"),
                mapped[0],
            )
            remote = T.decl_buffer((1,), "float32", scope="shared", data=remote_ptr)
            remote[0] = T.cast(10 + lane, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        output[cta] = shared[0]


@T.prim_func
def parallel_cluster_remote_shared_exchange(output: T.Buffer((2, 2, 2), "float32")):
    T.device_entry()
    cluster = T.cluster_id([2])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float32", scope="shared")
    if lane == 0:
        shared[0] = T.cast(cluster * 100 + cta * 10 + 1, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        peer = T.meta_var(1 - cta)
        mapped = T.alloc_local((1,), "uint64")
        T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(peer))
        remote_ptr: T.let[
            T.Var(name="parallel_remote_read_ptr", ty=PointerType(PrimType("float32"), "shared"))
        ] = T.reinterpret(
            PointerType(PrimType("float32"), "shared"),
            mapped[0],
        )
        remote = T.decl_buffer((1,), "float32", scope="shared", data=remote_ptr)
        output[cluster, cta, 0] = shared[0]
        output[cluster, cta, 1] = remote[0]


@T.prim_func
def mapped_remote_mbarrier_pointer(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            mapped = T.alloc_local((1,), "uint64")
            T.ptx.mapa.u64(mapped[0], barriers.ptr_to([0]), T.uint32(0))
            remote_ptr: T.let[
                T.Var(name="remote_barrier_ptr", ty=PointerType(PrimType("uint64"), "shared"))
            ] = T.reinterpret(
                PointerType(PrimType("uint64"), "shared"),
                mapped[0],
            )
            remote_barrier = T.decl_buffer((1,), "uint64", scope="shared", data=remote_ptr)
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(remote_barrier[0]))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.prim_func
def warp_pure_calls(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
    reduced: T.Buffer((32,), "uint32"),
    packed: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = source[lane]
    reverse: T.let = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), value, 31 - lane, 32, 32)
    magnitude = T.local_scalar("float32")
    reciprocal = T.local_scalar("float32")
    T.ptx.max.f32(magnitude, value, T.float32(0) - value)
    T.ptx.rcp.approx.ftz.f32(reciprocal, magnitude + T.float32(1))
    pair: T.let = T.cuda.make_float2(value, T.float32(2))
    multiplier: T.let = T.cuda.make_float2(T.float32(3), T.float32(4))
    product: T.let = T.cuda.fmul2_rn(pair, multiplier)
    output[lane] = reverse + reciprocal + T.cuda.float2_x(product)
    any_last: T.let = T.cuda.any_sync(T.uint32(0xFFFFFFFF), lane == 31)
    reduced[lane] = T.cuda.reduce_add_sync_u32(
        T.uint32(0xFFFFFFFF), T.cast(lane, "uint32")
    ) + T.cast(any_last, "uint32")
    packed[lane] = T.cuda.float22bfloat162_rn(value, reverse)


@T.prim_func
def native_varying_assert(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    with T.Assert(lane < 31, "lane must be below 31"):
        output[lane] = T.float32(1)


# -- kernel (verbatim from tests/numsim/support/remote_mbarrier.py) ----------


@T.prim_func
def mapped_remote_mbarrier_pointer_expect_tx(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            remote_ptr: T.let[
                T.Var(
                    name="remote_barrier_expect_tx_ptr",
                    ty=PointerType(PrimType("uint64"), "shared"),
                )
            ] = T.reinterpret(
                PointerType(PrimType("uint64"), "shared"),
                _mapa_u64(barriers.ptr_to([0]), 0),
            )
            remote_barrier = T.decl_buffer((1,), "uint64", scope="shared", data=remote_ptr)
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(remote_barrier[0]), 0)
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1



# -- tests ---------------------------------------------------------------------


def test_codegen_rejects_alpha_renamed_host_abi_with_the_same_structural_hash(tmp_path):
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_codegen_rejects_alpha_renamed_host_abi_with_the_same_structural_hash``.

    Legacy fed ``analyze(original)`` to ``emit_rust_module(..., renamed)`` and
    expected rejection. v2 has no separate spec/codegen step; the surviving
    contract is that two PrimFuncs equal up to alpha-renaming of the host ABI
    never share a compiled module: distinct cache keys (no cache hit in a
    shared cache dir), each module's host ABI keeps its own parameter name,
    and a binding under the other name is rejected (``v2.InputError``). Dropped:
    ``analyze``/``emit_rust_module`` and the "semantic/ABI manifest differs"
    message.
    """

    def make_func(parameter_name: str):
        buffer = tirx.decl_buffer((1,), "int32", name=parameter_name)
        return tirx.PrimFunc([buffer], tirx.Evaluate(tirx.IntImm("int32", 0)))

    original = make_func("original")
    renamed = make_func("renamed")
    assert structural_hash(original) == structural_hash(renamed)
    assert structural_equal(original, renamed)

    first = v2.transpile(original, cache_dir=tmp_path)
    second = v2.transpile(renamed, cache_dir=tmp_path)

    assert first.cache_key != second.cache_key
    assert not second.cache_hit
    assert [slot["name"] for slot in first.spec.kernels[0].host_abi] == ["original"]
    assert [slot["name"] for slot in second.spec.kernels[0].host_abi] == ["renamed"]
    with pytest.raises(v2.InputError):
        v2.Engine().run(second, {"original": np.zeros(1, dtype=np.int32)})


def test_warpgroup_relative_scope_ids_map_native_warp_coordinates():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_warpgroup_relative_scope_ids_map_native_warp_coordinates``.

    Dropped pin: ``result.stats["task_count"] == 8``.
    """

    output = np.zeros((2, 4, 32, 3), dtype=np.int32)
    expected = np.zeros_like(output)
    for warpgroup in range(2):
        for warp in range(4):
            for lane in range(32):
                expected[warpgroup, warp, lane] = (warpgroup, warp, warp * 32 + lane)

    result = _run(warpgroup_scope_coordinates, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_integer_elect_sync_condition_becomes_lane_mask():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_integer_elect_sync_condition_becomes_lane_mask``.

    Dropped pin: ``"integer_condition_mask" in module.rust_source``.
    """

    expected = np.full(32, 2, dtype=np.int32)
    expected[0] = 1

    result = _run(elect_sync_integer_branch, {"output": np.zeros(32, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_lane_add_uses_native_masked_warp_control():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_lane_add_uses_native_masked_warp_control``.

    Dropped: ``result.stats["task_count"]``, the ``module.rust_source`` pins
    (``warp_main``, ``parent_mask_branch``, ``WarpValue::from_fn``), and the
    second half that drove the legacy native artifact directly through
    ``prepare_bindings(...).to_payload`` and pinned its ``allocation_bytes``
    payload shape (legacy binding internals, no v2 counterpart).
    """

    left = np.arange(100, dtype=np.float32)
    right = np.linspace(0, 1, 100, dtype=np.float32)
    output = np.zeros(100, dtype=np.float32)

    result = _run(lane_add, {"left": left, "right": right, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], left + right)


def test_divergent_break_and_continue_use_loop_masks():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_divergent_break_and_continue_use_loop_masks``.

    Dropped pins: ``result.stats["task_count"]``, ``"live_mask_loop"`` and
    ``"ctx.set_active_mask(WarpMask::EMPTY)"`` in ``module.rust_source``.
    """

    output = np.zeros(128, dtype=np.float32)
    expected = np.zeros_like(output)
    live = np.ones(32, dtype=np.bool_)
    lanes = np.arange(32)
    for step in range(4):
        active = live.copy()
        active[lanes == step] = False
        breaking = active & (lanes < 4) & (step == 2)
        live[breaking] = False
        active[breaking] = False
        expected[lanes[active] * 4 + step] = 1

    result = _run(divergent_loop_control, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_divergent_while_preserves_lane_mask_and_scheduler_observation():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_divergent_while_preserves_lane_mask_and_scheduler_observation``.

    Dropped pin: ``result.stats["poll_order"] == [0]`` (legacy scheduler
    observation); the lane-mask outcome is kept.
    """

    expected = np.zeros(32, dtype=np.float32)
    expected[:16] = 2

    result = _run(divergent_while_control, {"output": np.zeros(32, dtype=np.float32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_empty_mask_after_continue_skips_mbarrier_wait():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_empty_mask_after_continue_skips_mbarrier_wait``.

    Dropped pin: ``"if !ctx.active_mask().is_empty()" in module.rust_source``.
    The run completing (no deadlock on the never-arrived barrier) with the
    output untouched is the observable.
    """

    output = np.zeros(32, dtype=np.int32)

    result = _run(mbarrier_wait_after_full_warp_continue, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)


def test_empty_mask_after_continue_skips_named_barrier_argument_evaluation():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_empty_mask_after_continue_skips_named_barrier_argument_evaluation``.

    Dropped pin: ``result.stats["task_count"] == 1``.
    """

    output = np.zeros(32, dtype=np.int32)

    result = _run(bar_sync_after_full_warp_continue, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)


def test_empty_mask_after_continue_skips_cta_sync():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_empty_mask_after_continue_skips_cta_sync``.

    Dropped pin: ``"if !ctx.active_mask().is_empty()" in module.rust_source``.
    """

    output = np.zeros(32, dtype=np.int32)

    result = _run(cta_sync_after_full_warp_continue, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)


def test_empty_mask_after_continue_skips_dynamic_for_argument_evaluation():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_empty_mask_after_continue_skips_dynamic_for_argument_evaluation``.

    Dropped pin: ``"if !parent_mask_loop" in module.rust_source``.
    """

    output = np.zeros(32, dtype=np.int32)

    result = _run(dynamic_for_after_full_warp_continue, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)


def test_scalar_expression_lowering_matches_floor_and_select_semantics():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_scalar_expression_lowering_matches_floor_and_select_semantics``.

    Dropped pins: ``floor_div_i64``, ``floor_mod_i64``, ``select_`` in
    ``module.rust_source``.
    """

    lanes = np.arange(32, dtype=np.int64)
    shifted = lanes - 17
    quotient = shifted // 5
    remainder = shifted % 5
    clamped = np.minimum(np.maximum(quotient, -2), 2)
    use_clamped = ((lanes < 5) & ~(lanes == 2)) | (lanes >= 29)
    expected = np.where(use_clamped, clamped, remainder).astype(np.float32)

    result = _run(scalar_expression_mix, {"output": np.zeros(32, dtype=np.float32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_composite_masks_cannot_reactivate_lanes_outside_parent_scope():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_composite_masks_cannot_reactivate_lanes_outside_parent_scope``.

    Dropped pins: ``"parent_mask_branch_"`` and ``"WarpMask::EMPTY"`` in
    ``module.rust_source``.
    """

    expected = np.zeros(32, dtype=np.int32)
    expected[:8] = 2
    expected[8:16] = 4

    result = _run(nested_mask_parent_scope, {"output": np.zeros(32, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_if_then_else_does_not_evaluate_an_unselected_buffer_load():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_if_then_else_does_not_evaluate_an_unselected_buffer_load``.

    Dropped pin: ``"ctx.set_active_mask(if_then_mask" in module.rust_source``.
    The run completing without an out-of-bounds stop on ``source[lane]`` for
    lanes 1..31 is the observable.
    """

    source = np.array([3.5], dtype=np.float32)
    expected = np.full(32, 7, dtype=np.float32)
    expected[0] = source[0]

    result = _run(guarded_if_then_else_load, {"source": source, "output": np.zeros(32, dtype=np.float32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_multidimensional_default_layout_lowers_to_physical_offsets():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_multidimensional_default_layout_lowers_to_physical_offsets``.

    Dropped pin: ``"T.TileLayout" not in module.rust_source``.
    """

    left = np.arange(15, dtype=np.float32).reshape(3, 5)
    right = np.linspace(0, 1, 15, dtype=np.float32).reshape(3, 5)
    output = np.zeros((3, 5), dtype=np.float32)

    result = _run(matrix_add_2d, {"left": left, "right": right, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], left + right)


def test_shared_allocations_alias_and_are_isolated_per_cta():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_shared_allocations_alias_and_are_isolated_per_cta``.

    Dropped pin: ``"runtime_buffer_shared(" in module.rust_source``.
    """

    expected = np.stack([np.arange(32, dtype=np.float32), 100 + np.arange(32, dtype=np.float32)])

    result = _run(shared_alias_per_cta, {"output": np.zeros((2, 32), dtype=np.float32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_local_register_arrays_are_isolated_per_lane():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_local_register_arrays_are_isolated_per_lane``.

    Dropped pin: ``"runtime_buffer_register(" in module.rust_source``.
    """

    expected = 2 * np.arange(32, dtype=np.float32) + 1

    result = _run(local_array_per_lane, {"output": np.zeros(32, dtype=np.float32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_runtime_shape_scalars_are_bound_from_consistent_buffer_descriptors():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_runtime_shape_scalars_are_bound_from_consistent_buffer_descriptors``.

    Dropped pins: ``"extract_shape_extent"`` in / ``"fn extract_shape_extent("``
    not in ``module.rust_source``.
    """

    source = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    output = np.zeros_like(source)

    result = _run(
        dynamic_rows,
        {"input_buffer": source, "output_buffer": output},
        outputs=("output_buffer",),
    )

    np.testing.assert_array_equal(result.outputs["output_buffer"], source + 2)


def test_runtime_shape_scalars_reject_disagreeing_bindings():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_runtime_shape_scalars_reject_disagreeing_bindings``.

    Dropped: ``match="runtime shape 'rows' disagrees"``. Kind: the input
    binding rejects the disagreeing shapes before anything runs
    (``v2.InputError``, a ``NumSimExecutionError`` as legacy expected).
    """

    module = v2.transpile(dynamic_rows)

    with pytest.raises(v2.InputError) as excinfo:
        v2.Engine().run(
            module,
            {
                "input_buffer": np.zeros((5, 32), dtype=np.float32),
                "output_buffer": np.zeros((4, 32), dtype=np.float32),
            },
        )
    assert isinstance(excinfo.value, NumSimExecutionError)
    assert not isinstance(excinfo.value, v2.MissingBindingsError)


def test_warp_local_raw_calls_execute_inside_one_warp_unit():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_warp_local_raw_calls_execute_inside_one_warp_unit``.

    Dropped pin: ``result.stats["task_count"] == 1``.
    """

    source = np.linspace(-2, 2, 32, dtype=np.float32)
    inputs = {
        "source": source,
        "output": np.zeros(32, dtype=np.float32),
        "reduced": np.zeros(32, dtype=np.uint32),
        "packed": np.zeros(32, dtype=np.uint32),
    }

    result = _run(warp_pure_calls, inputs)

    reverse = source[::-1].copy()
    expected = reverse + np.float32(1) / (np.abs(source) + np.float32(1)) + 3 * source
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=1e-06, atol=1e-06)
    np.testing.assert_array_equal(result.outputs["reduced"], np.full(32, 497, dtype=np.uint32))
    low = _encode_bf16(source).astype(np.uint32)
    high = _encode_bf16(reverse).astype(np.uint32) << np.uint32(16)
    np.testing.assert_array_equal(result.outputs["packed"], low | high)


def test_native_varying_assert_reports_failed_lane_mask():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_native_varying_assert_reports_failed_lane_mask``.

    Dropped: ``match="failed lanes=.*31"``. Kind: ``trap`` error whose
    structured ``lanes`` field is the failed-lane mask (only lane 31).
    """

    module = v2.transpile(native_varying_assert)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(32, dtype=np.float32)})
    stop = _assert_stop(excinfo, {"trap"})
    assert stop.get("lanes") == "WarpMask(0x80000000)", stop


def test_physical_mbarrier_reuses_phase_and_completes_numeric_transactions_directly():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_physical_mbarrier_reuses_phase_and_completes_numeric_transactions_directly``.

    Dropped pins: ``result.stats`` ``completed_task_count``,
    ``completion_operation_count`` and ``completion_pump_count``.
    """

    source = np.arange(4, dtype=np.float32)

    result = _run(mbarrier_phase_reuse, {"source": source, "output": np.zeros(2, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([1, 2], dtype=np.int32))


def test_physical_mbarrier_deadlock_reports_missing_lane_arrivals():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_physical_mbarrier_deadlock_reports_missing_lane_arrivals``.

    Dropped: the legacy message/payload pin ``"arrival_count=32/64"`` (the v2
    deadlock diagnostic carries no per-barrier arrival count) and the
    ``expect_harness_error`` fixture. Kind: ``deadlock`` error.
    """

    module = v2.transpile(mbarrier_missing_arrivals)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine(max_workers=1).run(module, {})
    _assert_stop(excinfo, {"deadlock"})


def test_mapa_remote_shared_writes_use_the_target_ctas_owner():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_mapa_remote_shared_writes_use_the_target_ctas_owner``.

    Dropped pins: ``result.stats["worker_count"] == 1`` and
    ``["scheduling_domain_count"] == 1``.
    """

    result = _run(remote_shared_write_ownership, {"output": np.zeros(2, dtype=np.float32)}, max_workers=8)

    np.testing.assert_array_equal(result.outputs["output"], np.array([10, 11], dtype=np.float32))


def test_parallel_clusters_keep_remote_shared_ownership_isolated():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_parallel_clusters_keep_remote_shared_ownership_isolated``.

    Dropped pins: ``result.stats`` ``task_count``, ``completed_task_count``,
    ``worker_count``, ``scheduling_domain_count``.
    """

    expected = np.array([[[1, 11], [11, 1]], [[101, 111], [111, 101]]], dtype=np.float32)

    result = _run(
        parallel_cluster_remote_shared_exchange,
        {"output": np.zeros((2, 2, 2), dtype=np.float32)},
        max_workers=2,
    )

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_local_mbarrier_arrive_rejects_mapped_remote_address():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_local_mbarrier_arrive_rejects_mapped_remote_address``.

    Dropped: ``match=r"local-form mbarrier\\.arrive.*global CTA 1.*remote
    global CTA 0"``. Kind: ``bad_address`` error.
    """

    module = v2.transpile(mapped_remote_mbarrier_pointer)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})
    _assert_stop(excinfo, {"bad_address"})


def test_local_mbarrier_arrive_expect_tx_rejects_mapped_remote_address():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_local_mbarrier_arrive_expect_tx_rejects_mapped_remote_address``.

    Dropped: ``match=r"local-form mbarrier\\.arrive.*global CTA 1.*remote
    global CTA 0"``. Kind: ``bad_address`` error.
    """

    module = v2.transpile(mapped_remote_mbarrier_pointer_expect_tx)

    with pytest.raises(v2.ExecutionError) as excinfo:
        v2.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})
    _assert_stop(excinfo, {"bad_address"})


def test_scoped_and_named_barriers_rendezvous_across_warps_and_ctas():
    """Port of ``tests/numsim/integration/test_artifact_build.py::test_scoped_and_named_barriers_rendezvous_across_warps_and_ctas``.

    Dropped pin: ``result.stats["completed_task_count"] == 8``.
    """

    cta = np.arange(2, dtype=np.int32)[:, None, None]
    warp = np.arange(4, dtype=np.int32)[None, :, None]
    lane = np.arange(32, dtype=np.int32)[None, None, :]
    expected = cta * 10000 + warp * 100 + lane

    result = _run(scoped_syncs, {"output": np.zeros((2, 4, 32), dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], expected)
