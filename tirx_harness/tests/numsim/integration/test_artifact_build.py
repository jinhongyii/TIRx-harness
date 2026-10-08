from __future__ import annotations

import json

import numpy as np
import pytest
from tvm.ir import load_json, save_json

from tirx_harness import numsim, racecheck
from tests.numsim.support.kernels import (
    bar_sync_after_full_warp_continue,
    bound_dynamic_cta_extent,
    bulk_shared_to_cluster_u64_addresses,
    compose_swizzle_alias,
    cta_sync_after_full_warp_continue,
    divergent_loop_control,
    divergent_while_control,
    dps_float_arithmetic,
    dynamic_for_after_full_warp_continue,
    dynamic_rows,
    elect_sync_integer_branch,
    extent_free_warp_id,
    guarded_if_then_else_load,
    guarded_select_load,
    lane_add,
    local_array_per_lane,
    mapped_remote_mbarrier_pointer,
    matrix_add_2d,
    mbarrier_missing_arrivals,
    mbarrier_phase_reuse,
    mbarrier_remote_cta,
    mbarrier_varying_uniform_phase,
    mbarrier_wait_after_full_warp_continue,
    native_varying_assert,
    nested_mask_parent_scope,
    no_op_kernel,
    overlapping_alias_write,
    parallel_cluster_remote_shared_exchange,
    physical_address_value,
    raw_scalar_call_mix,
    remote_shared_read_and_warp_reduce,
    remote_shared_write_ownership,
    scalar_buffer_types,
    scalar_expression_mix,
    scoped_syncs,
    shared_alias_per_cta,
    shared_uninitialized_read,
    warp_pure_calls,
    warpgroup_scope_coordinates,
)
from tests.numsim.support.remote_mbarrier import (
    mapped_remote_mbarrier_cluster_view,
    mapped_remote_mbarrier_pointer_expect_tx,
)
from tvm import tirx
from tvm_ffi import structural_equal, structural_hash
from tvm.script import tirx as T
from tvm.tirx.layout import ComposeLayout, S, TileLayout

_PADDED_COMPOSE_LAYOUT = ComposeLayout(0, 0, 0, TileLayout(S[(2, 2) : (4, 1)]))


@T.prim_func
def lazy_logical_buffer_load(source: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    and_value: T.let = (lane == 0) and (source[lane] == 5)
    or_value: T.let = (lane != 0) or (source[lane] == 5)
    output[lane] = T.cast(and_value, "int32") + T.cast(or_value, "int32") * 2


@T.prim_func
def lane_and_thread_scope_ids(output: T.Buffer((4, 32, 2), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    thread = T.thread_id([128])
    output[warp, lane, 0] = lane
    output[warp, lane, 1] = thread


@T.prim_func
def dps_pointer_offset(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.ptr_byte_offset(output.ptr_to([lane]), T.uint32(4), "float32")
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@T.prim_func
def dps_read_only_destination(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = output.access_ptr("r", offset=lane, extent=1)
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@T.prim_func
def dps_destination_after_access_extent(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    bounded = output.access_ptr("w", offset=lane, extent=1)
    destination = T.ptr_byte_offset(bounded, T.uint32(4), "float32")
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@T.prim_func
def remote_mbarrier_explicit_count(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 4)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            remote_barrier = T.alloc_local((1,), "uint64")
            T.ptx.mapa.shared__cluster.u64(remote_barrier[0], barriers.ptr_to([0]), T.uint32(0))
            T.ptx.mbarrier.arrive.b64(remote_barrier[0], T.uint32(4), pred=T.bool(True))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.prim_func
def padded_compose_alias(output: T.Buffer((6,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2, 2), "uint32", scope="shared", layout=_PADDED_COMPOSE_LAYOUT)
    dense = T.decl_buffer((6,), "uint32", data=shared.data, scope="shared")
    if lane == 0:
        dense[2] = 90
        dense[3] = 91
        shared[0, 0] = 10
        shared[0, 1] = 11
        shared[1, 0] = 12
        shared[1, 1] = 13
        for index in T.serial(6):
            output[index] = dense[index]


@T.prim_func
def cta_reductions_preserve_scratch(output: T.Buffer((3, 3), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    max_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    min_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    value = T.cast(warp * 100 + lane, "float32")
    sum_result = T.cuda.cta_sum(value, 2, sum_scratch.ptr_to([0]))
    max_result = T.cuda.cta_max(value, 2, max_scratch.ptr_to([0]))
    min_result = T.cuda.cta_min(value, 2, min_scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0, 0] = sum_result
        output[0, 1] = sum_scratch[0]
        output[0, 2] = sum_scratch[1]
        output[1, 0] = max_result
        output[1, 1] = max_scratch[0]
        output[1, 2] = max_scratch[1]
        output[2, 0] = min_result
        output[2, 1] = min_scratch[0]
        output[2, 2] = min_scratch[1]


@T.prim_func
def cuda_float_reduction_edge_bits(
    operands: T.Buffer((2,), "uint32"), output: T.Buffer((7,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    max_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    min_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    is_special = (warp == 0) and (lane == 1)
    nan_value = T.if_then_else(
        is_special, T.reinterpret("float32", T.uint32(0x7FC12345)), T.float32(0.0)
    )
    zero_value = T.if_then_else(is_special, T.float32(0.0), T.float32(-0.0))
    sum_result = T.cuda.cta_sum(nan_value, 2, sum_scratch.ptr_to([0]))
    max_result = T.cuda.cta_max(zero_value, 2, max_scratch.ptr_to([0]))
    min_result = T.cuda.cta_min(zero_value, 2, min_scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        lhs = T.reinterpret("float32", operands[0])
        rhs = T.reinterpret("float32", operands[1])
        output[0] = T.reinterpret("uint32", sum_result)
        output[1] = T.reinterpret("uint32", max_result)
        output[2] = T.reinterpret("uint32", min_result)
        output[3] = T.reinterpret("uint32", T.max(lhs, rhs))
        output[4] = T.reinterpret("uint32", T.max(rhs, lhs))
        output[5] = T.reinterpret("uint32", T.min(lhs, rhs))
        output[6] = T.reinterpret("uint32", T.min(rhs, lhs))


def _encode_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def test_empty_scope_id_names_do_not_alias_distinct_varying_builtins(tmp_path):
    payload = json.loads(save_json(lane_and_thread_scope_ids))
    for node in payload["nodes"]:
        if node["type"] == "tir.Var" and node["data"]["name"] in {"lane", "thread"}:
            node["data"]["name"] = ""
    kernel = load_json(json.dumps(payload))

    output = np.zeros((4, 32, 2), dtype=np.int32)
    result = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path),
        {"output": output},
    )

    lane = np.broadcast_to(np.arange(32, dtype=np.int32), (4, 32))
    thread = np.arange(128, dtype=np.int32).reshape(4, 32)
    np.testing.assert_array_equal(result.outputs["output"][..., 0], lane)
    np.testing.assert_array_equal(result.outputs["output"][..., 1], thread)


def test_extent_free_warp_id_uses_flat_native_coordinate(tmp_path):
    output = np.zeros((2, 32), dtype=np.int32)
    expected = np.broadcast_to(np.arange(2, dtype=np.int32)[:, None], output.shape)

    module = numsim.transpile(extent_free_warp_id, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_single_flat_scope_id_ignores_bound_dynamic_launch_extent(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(bound_dynamic_cta_extent, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([1, 2], dtype=np.int32))


def test_typed_raw_scalar_calls_execute_with_warp_semantics(tmp_path):
    source = 1.0 + np.arange(32, dtype=np.float32) / np.float32(32)
    output = np.zeros(32, dtype=np.float32)
    lanes = np.arange(32, dtype=np.uint32)
    low_nibble = lanes & np.uint32(15)
    shifted = (low_nibble << np.uint32(1)) | (low_nibble >> np.uint32(1))
    mixed = shifted ^ np.uint32(3)
    inverted = (~low_nibble) & np.uint32(15)
    selected = source.copy()
    selected[[0, 31]] = selected[[0, 31]] * np.float32(2) + np.float32(1)
    elected = np.zeros(32, dtype=np.float32)
    elected[0] = 1
    expected = (
        np.exp2(np.log(selected)).astype(np.float32)
        + np.float32(1) / np.sqrt(selected + np.float32(4))
        + (mixed + inverted).astype(np.float32) * np.float32(0.001)
        + elected * np.float32(0.01)
    )

    module = numsim.transpile(raw_scalar_call_mix, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_allclose(result.outputs["output"], expected, rtol=2e-06, atol=2e-06)


def test_mbarrier_phase_remains_lane_vector_inside_the_engine(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mbarrier_varying_uniform_phase, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([0, 1], dtype=np.int32))


def test_parameter_names_and_explicit_outputs_are_respected(tmp_path):
    left = np.arange(100, dtype=np.float32)
    right = np.ones(100, dtype=np.float32)
    output = np.zeros(100, dtype=np.float32)

    module = numsim.transpile(lane_add, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"left": left, "right": right, "output": output},
        outputs=("output",),
    )

    assert set(result.outputs) == {"output"}
    np.testing.assert_array_equal(result.outputs["output"], left + right)


def test_shared_uninitialized_read_is_zero_filled_and_requires_review(tmp_path):
    module = numsim.transpile(shared_uninitialized_read, cache_dir=tmp_path)
    output = np.full(32, np.float32(7), dtype=np.float32)

    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(32, dtype=np.float32))
    assert result.verdict == "review"
    assert result.diagnostics
    assert {item["status"] for item in result.diagnostics} == {"review"}
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}

    report = numsim.compare(result, {"output": np.ones(32, dtype=np.float32)})
    assert report.verdict == "error"
    assert not report.ok


def test_scalar_buffer_dtypes_use_physical_little_endian_storage(tmp_path):
    input_i32 = np.arange(-16, 16, dtype=np.int32)
    output_i32 = np.zeros(32, dtype=np.int32)
    input_u64 = np.arange(32, dtype=np.uint64) * np.uint64(1000)
    output_u64 = np.zeros(32, dtype=np.uint64)
    input_f16 = np.linspace(-4, 4, 32, dtype=np.float16)
    output_f16 = np.zeros(32, dtype=np.float16)
    input_bf16_bits = _encode_bf16(np.linspace(-3, 3, 32, dtype=np.float32))
    output_bf16_bits = np.zeros(32, dtype=np.uint16)
    input_bool = np.arange(32) % 3 == 0
    output_bool = np.zeros(32, dtype=np.bool_)

    input_bf16 = input_bf16_bits
    output_bf16 = output_bf16_bits
    module = numsim.transpile(scalar_buffer_types, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_i32": input_i32,
            "output_i32": output_i32,
            "input_u64": input_u64,
            "output_u64": output_u64,
            "input_f16": input_f16,
            "output_f16": output_f16,
            "input_bf16": input_bf16,
            "output_bf16": output_bf16,
            "input_bool": input_bool,
            "output_bool": output_bool,
        },
    )

    np.testing.assert_array_equal(result.outputs["output_i32"], input_i32 + 7)
    np.testing.assert_array_equal(result.outputs["output_u64"], input_u64 + 11)
    expected_f16 = (input_f16.astype(np.float32) + 0.5).astype(np.float16)
    np.testing.assert_array_equal(result.outputs["output_f16"], expected_f16)
    expected_bf16 = _encode_bf16(_decode_bf16(input_bf16_bits) + 0.5)
    np.testing.assert_array_equal(result.outputs["output_bf16"], expected_bf16)
    np.testing.assert_array_equal(result.outputs["output_bool"], input_bool)


def test_compose_swizzle_layout_resolves_to_physical_alias_bytes(tmp_path):
    output = np.zeros(32, dtype=np.float32)

    module = numsim.transpile(compose_swizzle_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 33, dtype=np.float32))


def test_padded_compose_layout_allocates_its_physical_span_for_aliases(tmp_path):
    output = np.zeros(6, dtype=np.uint32)

    module = numsim.transpile(padded_compose_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.array([10, 11, 90, 91, 12, 13], dtype=np.uint32)
    )


def test_cta_reductions_preserve_each_warp_partial_in_scratch(tmp_path):
    output = np.zeros((3, 3), dtype=np.float32)

    module = numsim.transpile(cta_reductions_preserve_scratch, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array(
            [[4192.0, 4192.0, 3696.0], [131.0, 131.0, 131.0], [0.0, 0.0, 100.0]],
            dtype=np.float32,
        ),
    )


def test_dps_float_arithmetic_writes_through_lane_private_physical_pointers(tmp_path):
    output = np.zeros((32, 5), dtype=np.float32)

    module = numsim.transpile(dps_float_arithmetic, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    lane = np.arange(32, dtype=np.float32)
    expected = np.stack(
        [lane * 2 + 1, lane * 2 + 4, (lane + 1) * 3 + 5, lane + 2, lane + 4], axis=1
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_dps_writes_through_the_full_physical_pointer(tmp_path):
    output = np.full(33, -1, dtype=np.float32)
    module = numsim.transpile(dps_pointer_offset, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": output})

    expected = np.concatenate(
        (np.array([-1], dtype=np.float32), np.arange(100, 132, dtype=np.float32))
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_remote_mbarrier_arrive_targets_the_other_ctas_physical_slot(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mbarrier_remote_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_remote_mbarrier_explicit_count_completes_the_target_phase(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(remote_mbarrier_explicit_count, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_cluster_mbarrier_arrive_recovers_target_from_pointer_derived_view(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mapped_remote_mbarrier_cluster_view, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_mapa_remote_shared_reads_and_subgroup_reductions_run_inside_one_warp(tmp_path):
    output = np.zeros((2, 3, 32), dtype=np.float32)
    expected = np.zeros_like(output)
    expected[:, 0, :2] = 3
    expected[:, 1, :2] = 2
    expected[:, 2, :2] = 1

    module = numsim.transpile(remote_shared_read_and_warp_reduce, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_bulk_shared_to_cluster_consumes_public_u64_mapa_addresses(tmp_path):
    module = numsim.transpile(bulk_shared_to_cluster_u64_addresses, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((2, 4), dtype=np.float32)})

    expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


