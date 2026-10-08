"""v2 port of the legacy Synccheck protocol-bearing payload runtime test.

Legacy: tests/analysis_tools/synccheck/runtime/test_device_payload_ops.py::test_payload_runtime
analyzed and transpiled all kernels below into one _analysis_capable module,
checked that each kernel carries the call ops _KERNEL_BY_OP assigns to it
(legacy ``analyze`` source map via ``tests.numsim.support.manifest.call_op_names``)
and ran Engine.run_synccheck_phase per kernel.

The v2 copy runs v2.synccheck on each kernel with the legacy coverage bounds
and resource limits and keeps the verdict contract: phase name, no execution
error, no findings, verdict clean, no incomplete and
coverage.eligible_for_clean. Dropped: the op-ownership check (legacy
``analyze`` source map), the legacy-vs-analyze manifest equality,
stats["task_count"] > 0, stats["completed_task_count"] == stats["task_count"]
and search["algorithm"] == "fixed_sync_state" (legacy scheduler/explorer
internals), plus the legacy-only engine knobs (max_workers, native_loop_*,
max_polls, max_transitions) and _analysis_capable. Kernels from legacy test
modules are copied verbatim (each block is marked with its source path);
kernels from surviving fixture modules are imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.microtests.cases.mma_sync import (
    make_dense_f16_case,
    make_sparse_f16_case,
    raw_dense_f16_f32_k16,
    raw_sparse_f16_f32_k32,
)
from tests.numsim.microtests.cases.tcgen05_advanced_mma import tcgen05_block_scaled_mxf4
from tests.numsim.microtests.cases.tcgen05_lifecycle_ldst import (
    tcgen05_bf16_mma,
    tcgen05_cp_warpx4,
)
from tests.numsim.support.kernels import raw_tma_roundtrip
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


# --- copied from tests/analysis_tools/synccheck/runtime/test_device_payload_ops.py ---

@T.prim_func
def dps_f32_protocol_ops(
    output_f32: T.Buffer((32, 4), "float32"),
    output_f32x2: T.Buffer((32, 4), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tiny: T.let = T.cuda.uint_as_float(T.uint32(1))
    increment: T.let = T.float32(2**-25)
    nan: T.let = T.cuda.uint_as_float(T.uint32(0x7FC00001))

    T.ptx.add.rn.f32(output_f32[lane, 0], T.float32(1), increment)
    T.ptx.sub.rn.f32(output_f32[lane, 1], T.float32(-1), increment)
    T.ptx.mul.rn.f32(output_f32[lane, 2], tiny, T.float32(1))
    T.ptx.fma.rn.f32(output_f32[lane, 3], nan, T.float32(1), T.float32(0))

    packed_lhs: T.let = T.cuda.make_float2(T.float32(1), tiny)
    packed_rhs: T.let = T.cuda.make_float2(increment, T.float32(1))
    packed_addend: T.let = T.cuda.make_float2(tiny, tiny)
    T.ptx.add.rn.f32x2(output_f32x2[lane, 0], packed_lhs, packed_rhs)
    T.ptx.sub.rn.f32x2(output_f32x2[lane, 1], packed_lhs, packed_rhs)
    T.ptx.mul.rn.f32x2(output_f32x2[lane, 2], packed_lhs, packed_rhs)
    T.ptx.fma.rn.f32x2(
        output_f32x2[lane, 3],
        packed_lhs,
        packed_rhs,
        packed_addend,
    )


@T.prim_func
def dps_f64_protocol_ops(output: T.Buffer((32, 4), "float64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = T.cast(lane, "float64")

    T.ptx.add.rn.f64(output[lane, 0], value, T.float64(2))
    T.ptx.sub.rn.f64(output[lane, 1], value, T.float64(2))
    T.ptx.mul.rn.f64(output[lane, 2], value, T.float64(2))
    T.ptx.fma.rn.f64(output[lane, 3], value, T.float64(2), T.float64(1))


# --- copied from tests/numsim/runtime/test_matrix_instruction_codegen.py ---

@T.prim_func
def ptx_mma_legacy_f16_m16n8k16(
    a: T.Buffer((16, 16), "float16"),
    b: T.Buffer((16, 8), "float16"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((8,), "float16")
    b_regs = T.alloc_local((4,), "float16")
    accumulator = T.alloc_local((4,), "float32")

    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 2 + (register // 2) * 8)
        a_regs[register * 2] = a[row, col]
        a_regs[register * 2 + 1] = a[row, col + 1]
    for register in T.unroll(2):
        row = T.meta_var(thread * 2 + register * 8)
        b_regs[register * 2] = b[row, group]
        b_regs[register * 2 + 1] = b[row + 1, group]
    accumulator[0] = c[group, thread * 2]
    accumulator[1] = c[group, thread * 2 + 1]
    accumulator[2] = c[group + 8, thread * 2]
    accumulator[3] = c[group + 8, thread * 2 + 1]

    T.ptx_legacy.mma(
        "m16n8k16",
        "row",
        "col",
        "float16",
        "float16",
        "float32",
        a_regs.data,
        0,
        b_regs.data,
        0,
        accumulator.data,
        0,
        False,
        dtype="float32",
    )

    output[group, thread * 2] = accumulator[0]
    output[group, thread * 2 + 1] = accumulator[1]
    output[group + 8, thread * 2] = accumulator[2]
    output[group + 8, thread * 2 + 1] = accumulator[3]


# --- copied from tests/numsim/runtime/test_matrix_instruction_codegen.py ---

@T.prim_func
def mma_fragment_fill_and_store(
    filled: T.Buffer((2, 32, 8), "float32"),
    stored: T.Buffer((2, 16, 16), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current_fragment = T.alloc_local((8,), "float32")
    legacy_fragment = T.alloc_local((8,), "float32")
    current_fill = T.alloc_local((8,), "float32")
    legacy_fill = T.alloc_local((8,), "float32")
    for local_id in T.serial(8):
        current_fragment[local_id] = T.cast(lane * 100 + local_id, "float32")
        legacy_fragment[local_id] = T.cast(10000 + lane * 100 + local_id, "float32")
        current_fill[local_id] = T.float32(1)
        legacy_fill[local_id] = T.float32(2)

    T.evaluate(T.cuda.mma_fill(8, current_fill.data, 0, dtype="float32"))
    T.evaluate(T.cuda.mma_fill_legacy(8, legacy_fill.data, 0, dtype="float32"))
    T.evaluate(
        T.cuda.mma_store(
            16,
            16,
            stored.ptr_to([0, 0, 0]),
            current_fragment.data,
            0,
            16,
            dtype="float32",
        )
    )
    T.evaluate(
        T.cuda.mma_store_legacy(
            16,
            16,
            stored.ptr_to([1, 0, 0]),
            legacy_fragment.data,
            0,
            16,
            dtype="float32",
        )
    )
    for local_id in T.serial(8):
        filled[0, lane, local_id] = current_fill[local_id]
        filled[1, lane, local_id] = legacy_fill[local_id]


# --- copied from tests/numsim/runtime/test_memory_ops.py ---

@T.prim_func
def raw_tma_gather4_bar_address(input_map: T.TensorMap(), output: T.Buffer((4, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 4), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx[
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1"
        ](
            T.address_of(shared[0, 0]),
            T.address_of(input_map),
            0,
            0,
            1,
            2,
            3,
            T.cuda.cvta_generic_to_shared(T.address_of(barrier[0])),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 64)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane // 4, lane % 4] = shared[lane // 4, lane % 4]


# --- copied from tests/numsim/runtime/test_memory_ops.py ---

@T.prim_func
def raw_tma_reduce_add(source: T.Buffer((4,), "float32"), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group"](
            T.address_of(output_map), 0, T.address_of(shared[0])
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


# --- copied from tests/numsim/runtime/test_memory_ops.py ---

@T.prim_func
def raw_tma_prefetch(input_map: T.TensorMap(), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile"](T.address_of(input_map), 0, 0)
        output[0] = 1


# --- copied from tests/numsim/runtime/test_non_tensor_bulk_forms.py ---

@T.prim_func
def raw_bulk_prefetch(
    source: T.Buffer((64,), "uint8"), num_bytes: T.uint32, output: T.Buffer((32,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.L2.global"](source.ptr_to([16]), num_bytes)
    output[lane] = source[lane + 16]


# --- copied from tests/numsim/runtime/test_ptx_register_bits.py ---

@T.prim_func
def ptx_predicate_data_path(
    lhs_f32: T.Buffer((32,), "float32"),
    lhs_u32: T.Buffer((32,), "uint32"),
    selected_u32: T.Buffer((32,), "uint32"),
    selected_f32: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    p_float = T.alloc_local((1,), "uint32")
    p_integer = T.alloc_local((1,), "uint32")
    p_both = T.alloc_local((1,), "uint32")
    T.ptx.setp.le.f32(p_float[0], lhs_f32[lane], T.float32(0))
    T.ptx.setp.ne.u32(p_integer[0], lhs_u32[lane], T.uint32(0))
    T.ptx.and_.pred(p_both[0], T.ptx.pred(p_float[0]), T.ptx.pred(p_integer[0]))
    T.ptx.selp.u32(selected_u32[lane], lhs_u32[lane], T.uint32(99), T.ptx.pred(p_both[0]))
    T.ptx.selp.f32(selected_f32[lane], lhs_f32[lane], T.float32(7), T.ptx.pred(p_both[0]))


# --- copied from tests/numsim/runtime/test_ptx_register_bits.py ---

@T.prim_func
def ptx_selp_b16_bits(
    predicate: T.Buffer((32,), "uint32"),
    on_true: T.Buffer((32,), "uint16"),
    on_false: T.Buffer((32,), "uint16"),
    selected: T.Buffer((32,), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.selp.b16(selected[lane], on_true[lane], on_false[lane], T.ptx.pred(predicate[lane]))


# --- copied from tests/numsim/runtime/test_ptx_register_bits.py ---

@T.prim_func
def ptx_half_abs_and_setp(
    packed: T.Buffer((32,), "uint32"),
    lhs: T.Buffer((32,), "uint16"),
    rhs: T.Buffer((32,), "uint16"),
    absolute: T.Buffer((32,), "uint32"),
    greater: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.abs.f16x2(absolute[lane], packed[lane])
    T.ptx.setp.gt.f16(greater[lane], lhs[lane], rhs[lane])


# --- copied from tests/numsim/runtime/test_ptx_register_bits.py ---

@T.prim_func
def ptx_typed_move_and_b16_pack(
    low: T.Buffer((32,), "uint16"),
    high: T.Buffer((32,), "uint16"),
    source_i32: T.Buffer((32,), "int32"),
    packed: T.Buffer((32,), "uint32"),
    unpacked_low: T.Buffer((32,), "uint16"),
    unpacked_high: T.Buffer((32,), "uint16"),
    moved_i32: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mov.b32(packed[lane], low[lane], high[lane])
    T.ptx.mov.b32(unpacked_low[lane], unpacked_high[lane], packed[lane])
    T.ptx.mov.s32(moved_i32[lane], source_i32[lane])


# --- copied from tests/numsim/runtime/test_scalar_control.py ---

@T.prim_func
def pointer_conversions_and_descriptor(
    source: T.Buffer((32, 8), "float32"),
    half: T.Buffer((32, 8), "float16"),
    roundtrip: T.Buffer((32, 8), "float32"),
    descriptor: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.cuda.float8tohalf8(T.address_of(source[lane, 0]), T.address_of(half[lane, 0]))
    T.cuda.half8tofloat8(T.address_of(half[lane, 0]), T.address_of(roundtrip[lane, 0]))
    T.cuda.float22half2(T.address_of(half[lane, 0]), T.address_of(source[lane, 0]))
    T.cuda.runtime_instr_desc(T.address_of(descriptor[lane]), lane % 4)


# --- copied from tests/numsim/runtime/test_memory_ops.py ---

def _tensor_map(
    array: np.ndarray,
    *,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
) -> np.ndarray:
    return TensorMap(
        base=array,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
    ).numpy()


_KERNELS = (
    pointer_conversions_and_descriptor,
    dps_f32_protocol_ops,
    dps_f64_protocol_ops,
    ptx_predicate_data_path,
    ptx_selp_b16_bits,
    ptx_half_abs_and_setp,
    ptx_typed_move_and_b16_pack,
    mma_fragment_fill_and_store,
    raw_dense_f16_f32_k16,
    ptx_mma_legacy_f16_m16n8k16,
    raw_sparse_f16_f32_k32,
    tcgen05_cp_warpx4,
    tcgen05_bf16_mma,
    tcgen05_block_scaled_mxf4,
    raw_tma_roundtrip,
    raw_tma_prefetch,
    raw_tma_gather4_bar_address,
    raw_tma_reduce_add,
    raw_bulk_prefetch,
)


def _kernel_name(kernel: Any) -> str:
    return getattr(kernel, "__name__", None) or str(kernel.attrs["global_symbol"])


_BY_NAME = {_kernel_name(kernel): kernel for kernel in _KERNELS}


# --- copied from tests/analysis_tools/synccheck/runtime/test_device_payload_ops.py ---

def _arguments_by_kernel() -> dict[str, dict[str, Any]]:
    tma_roundtrip_source = np.arange(12, dtype=np.float32).reshape(3, 4)
    tma_roundtrip_output = np.zeros_like(tma_roundtrip_source)
    gather_source = np.arange(16, dtype=np.float32).reshape(4, 4)
    reduce_output = np.zeros(4, dtype=np.float32)
    return {
        "pointer_conversions_and_descriptor": {
            "source": np.arange(32 * 8, dtype=np.float32).reshape(32, 8),
            "half": np.zeros((32, 8), dtype=np.float16),
            "roundtrip": np.zeros((32, 8), dtype=np.float32),
            "descriptor": np.zeros(32, dtype=np.uint32),
        },
        "dps_f32_protocol_ops": {
            "output_f32": np.zeros((32, 4), dtype=np.float32),
            "output_f32x2": np.zeros((32, 4), dtype=np.uint64),
        },
        "dps_f64_protocol_ops": {"output": np.zeros((32, 4), dtype=np.float64)},
        "ptx_predicate_data_path": {
            "lhs_f32": np.arange(32, dtype=np.float32) - np.float32(16),
            "lhs_u32": np.arange(32, dtype=np.uint32) % np.uint32(3),
            "selected_u32": np.zeros(32, dtype=np.uint32),
            "selected_f32": np.zeros(32, dtype=np.float32),
        },
        "ptx_selp_b16_bits": {
            "predicate": np.arange(32, dtype=np.uint32) & np.uint32(1),
            "on_true": np.arange(32, dtype=np.uint16) ^ np.uint16(0x8000),
            "on_false": np.arange(32, dtype=np.uint16) ^ np.uint16(0x7C00),
            "selected": np.zeros(32, dtype=np.uint16),
        },
        "ptx_half_abs_and_setp": {
            "packed": np.arange(32, dtype=np.uint32) ^ np.uint32(0x80008000),
            "lhs": np.arange(32, dtype=np.float16).view(np.uint16),
            "rhs": (np.arange(32, dtype=np.float16) - np.float16(1)).view(np.uint16),
            "absolute": np.zeros(32, dtype=np.uint32),
            "greater": np.zeros(32, dtype=np.uint32),
        },
        "ptx_typed_move_and_b16_pack": {
            "low": np.arange(32, dtype=np.uint16),
            "high": np.uint16(0xFFFF) - np.arange(32, dtype=np.uint16),
            "source_i32": np.arange(32, dtype=np.int32) - np.int32(16),
            "packed": np.zeros(32, dtype=np.uint32),
            "unpacked_low": np.zeros(32, dtype=np.uint16),
            "unpacked_high": np.zeros(32, dtype=np.uint16),
            "moved_i32": np.zeros(32, dtype=np.int32),
        },
        "mma_fragment_fill_and_store": {
            "filled": np.zeros((2, 32, 8), dtype=np.float32),
            "stored": np.zeros((2, 16, 16), dtype=np.float32),
        },
        "raw_dense_f16_f32_k16": dict(make_dense_f16_case()),
        "ptx_mma_legacy_f16_m16n8k16": {
            "a": np.zeros((16, 16), dtype=np.float16),
            "b": np.zeros((16, 8), dtype=np.float16),
            "c": np.zeros((16, 8), dtype=np.float32),
            "output": np.zeros((16, 8), dtype=np.float32),
        },
        "raw_sparse_f16_f32_k32": dict(make_sparse_f16_case()),
        "tcgen05_cp_warpx4": {
            "source": np.arange(32 * 4, dtype=np.uint32).reshape(32, 4),
            "output": np.zeros((4, 32, 4), dtype=np.uint32),
        },
        "tcgen05_bf16_mma": {"output": np.zeros((4, 32, 4), dtype=np.float32)},
        "tcgen05_block_scaled_mxf4": {"output": np.zeros((4, 32, 16), dtype=np.float32)},
        "raw_tma_roundtrip": {
            "input_map": _tensor_map(
                tma_roundtrip_source,
                global_shape=(4, 3),
                global_strides=(16,),
                box_shape=(4, 3),
            ),
            "output_map": _tensor_map(
                tma_roundtrip_output,
                global_shape=(4, 3),
                global_strides=(16,),
                box_shape=(4, 3),
            ),
        },
        "raw_tma_prefetch": {
            "input_map": _tensor_map(
                gather_source,
                global_shape=(4, 4),
                global_strides=(16,),
                box_shape=(4, 1),
            ),
            "output": np.zeros(1, dtype=np.int32),
        },
        "raw_tma_gather4_bar_address": {
            "input_map": _tensor_map(
                gather_source,
                global_shape=(4, 4),
                global_strides=(16,),
                box_shape=(4, 1),
            ),
            "output": np.zeros((4, 4), dtype=np.float32),
        },
        "raw_tma_reduce_add": {
            "source": np.arange(4, dtype=np.float32),
            "output_map": _tensor_map(
                reduce_output,
                global_shape=(4,),
                global_strides=(),
                box_shape=(4,),
            ),
        },
        "raw_bulk_prefetch": {
            "source": np.arange(64, dtype=np.uint8) ^ np.uint8(0xA5),
            "num_bytes": np.uint32(32),
            "output": np.zeros(32, dtype=np.uint8),
        },
    }


_LEGACY_SURFACE = (
    "v2 transpile rejects the legacy {ops} surface (UnsupportedTIRxError; "
    "synccheck verdict incomplete, native_frontend_unsupported); legacy accepted it "
    "and reported clean. lowering-inventory.md E.3 lists `ptx_legacy.*` / `mma_*` "
    "as out of scope, but no delta row rules it"
)

_GAPS = {
    "ptx_mma_legacy_f16_m16n8k16": v2_gap(_LEGACY_SURFACE.format(ops="tirx.ptx_legacy.mma")),
}


@pytest.mark.parametrize(
    "kernel_name",
    [pytest.param(name, marks=(_GAPS[name],) if name in _GAPS else (), id=name) for name in _BY_NAME],
)
def test_payload_runtime(kernel_name):
    """Port of tests/analysis_tools/synccheck/runtime/test_device_payload_ops.py::test_payload_runtime.

    Dropped: the op-ownership check over the legacy ``analyze`` source map,
    stats task counts and search["algorithm"] (see the module docstring).
    """

    report = v2.synccheck(
        _BY_NAME[kernel_name],
        _arguments_by_kernel()[kernel_name],
        coverage_bounds=v2.CoverageBounds(0, 0),
        resource_limits=v2.ResourceLimits(
            max_schedules=100,
            max_backtrack_nodes=100_000,
            max_events_per_run=100_000,
            max_total_events=1_000_000,
            max_loop_steps=1_000_000,
            max_wall_time_ms=30_000,
            max_diagnostic_bytes=1_000_000,
        ),
    )
    assert len(report.phases) == 1
    result = report.phases[0].to_dict()
    assert result["phase"]["name"] == kernel_name
    assert result["execution_error"] is None, report.format()
    assert result["findings"] == [], report.format()
    assert result["verdict"] == "clean", report.format()
    assert result["incomplete"] == [], report.format()
    assert result["coverage"]["eligible_for_clean"] is True
