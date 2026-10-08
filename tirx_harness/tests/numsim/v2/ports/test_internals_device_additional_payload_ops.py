"""v2 port of the legacy Synccheck exact-op payload runtime test.

Legacy: tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime
transpiled all kernels below into one _analysis_capable module and ran
Engine.run_synccheck_phase per kernel through the helper
tests/analysis_tools/synccheck/runtime/test_device_payload_ops.py::_assert_payload_runtime.

The v2 copy runs v2.synccheck on each kernel with the legacy coverage
bounds and resource limits and keeps the verdict contract: phase name,
no execution error, no findings, verdict clean, no incomplete and
coverage.eligible_for_clean. Dropped pins: stats["task_count"] > 0,
stats["completed_task_count"] == stats["task_count"] and
search["algorithm"] == "fixed_sync_state" (legacy scheduler/explorer
internals), plus the legacy-only engine knobs (max_workers,
native_loop_*, max_polls, max_transitions) and
_analysis_capable. Kernels are copied verbatim from their legacy modules
(each block is marked with its source path).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tmem_datapath_layout

from tests.numsim.support.kernels import tcgen_commit_runtime_multicast
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


# --- copied from tests/numsim/integration/test_direct_memory_artifact.py ---

@T.prim_func
def raw_cp_async_zero_fill(
    source: T.Buffer((128,), "float32"), output: T.Buffer((128,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")
    T.evaluate(
        T.ptx["cp.async.cg.shared.global.L2::128B"](
            T.address_of(shared[lane * 4]),
            T.address_of(source[lane * 4]),
            16,
            T.cast(T.if_then_else(lane < 16, 16, 0), "uint32"),
        )
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.prim_func
def pass_emitted_cp_async_raw(
    source: T.Buffer((128,), "float32"), output: T.Buffer((128,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")
    T.ptx["cp.async.cg.shared.global"](shared.ptr_to([lane * 4]), source.ptr_to([lane * 4]), 16)
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


# --- copied from tests/numsim/integration/test_packed_ptx_cvt_and_return.py ---

@T.prim_func
def packed_ptx_cvt(
    high: T.Buffer((8,), "float32"),
    low: T.Buffer((8,), "float32"),
    e4m3x2: T.Buffer((8,), "uint16"),
    packed_bf16: T.Buffer((32,), "uint32"),
    packed_e8m0: T.Buffer((32,), "uint16"),
    unpacked_e8m0: T.Buffer((32,), "uint32"),
    unpacked_e4m3_bf16: T.Buffer((32,), "uint32"),
    unpacked_e4m3_f16: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    index: T.let = lane % 8
    packed = T.local_scalar("uint16")
    T.ptx.cvt.rn.bf16x2.f32(packed_bf16[lane], high[index], low[index])
    T.ptx.cvt.rz.ue8m0x2.f32(packed, high[index], low[index])
    packed_e8m0[lane] = packed
    T.ptx.cvt.rn.bf16x2.ue8m0x2(unpacked_e8m0[lane], packed)
    T.ptx.cvt.rn.bf16x2.e4m3x2(unpacked_e4m3_bf16[lane], e4m3x2[index])
    T.ptx.cvt.rn.f16x2.e4m3x2(unpacked_e4m3_f16[lane], e4m3x2[index])


# --- copied from tests/numsim/microtests/cases/ptx_cvt_fp8.py ---

@T.prim_func
def packed_fp8_cvt_sm100_forms(
    f32_source: T.Buffer((256,), "float32"),
    f16x2_source: T.Buffer((256,), "uint32"),
    f8x2_source: T.Buffer((256,), "uint16"),
    pack_from_f32: T.Buffer((4, 256), "uint16"),
    pack_from_f16x2: T.Buffer((4, 256), "uint16"),
    unpack_to_f16x2: T.Buffer((4, 256), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        index: T.let = step * 32 + lane
        following: T.let = (index + 1) % 256
        high: T.let = f32_source[index]
        low: T.let = f32_source[following]
        packed_f16x2: T.let = f16x2_source[index]
        packed_f8x2: T.let = f8x2_source[index]

        # cvt.rn.satfinite{.relu}.f8x2type.f32
        T.ptx["cvt.rn.satfinite.e4m3x2.f32"](pack_from_f32[0, index], high, low)
        T.ptx["cvt.rn.satfinite.relu.e4m3x2.f32"](pack_from_f32[1, index], high, low)
        T.ptx["cvt.rn.satfinite.e5m2x2.f32"](pack_from_f32[2, index], high, low)
        T.ptx["cvt.rn.satfinite.relu.e5m2x2.f32"](pack_from_f32[3, index], high, low)

        # cvt.rn.satfinite{.relu}.f8x2type.f16x2
        T.ptx["cvt.rn.satfinite.e4m3x2.f16x2"](pack_from_f16x2[0, index], packed_f16x2)
        T.ptx["cvt.rn.satfinite.relu.e4m3x2.f16x2"](pack_from_f16x2[1, index], packed_f16x2)
        T.ptx["cvt.rn.satfinite.e5m2x2.f16x2"](pack_from_f16x2[2, index], packed_f16x2)
        T.ptx["cvt.rn.satfinite.relu.e5m2x2.f16x2"](pack_from_f16x2[3, index], packed_f16x2)

        # cvt.rn{.relu}.f16x2.f8x2type
        T.ptx["cvt.rn.f16x2.e4m3x2"](unpack_to_f16x2[0, index], packed_f8x2)
        T.ptx["cvt.rn.relu.f16x2.e4m3x2"](unpack_to_f16x2[1, index], packed_f8x2)
        T.ptx["cvt.rn.f16x2.e5m2x2"](unpack_to_f16x2[2, index], packed_f8x2)
        T.ptx["cvt.rn.relu.f16x2.e5m2x2"](unpack_to_f16x2[3, index], packed_f8x2)


# --- copied from tests/numsim/microtests/cases/ptx_cvt_narrow.py ---

@T.prim_func
def narrow_cvt_sm100_forms(
    f32_source: T.Buffer((256,), "float32"),
    f16x2_source: T.Buffer((256,), "uint32"),
    bf16x2_source: T.Buffer((256,), "uint32"),
    f4x2_source: T.Buffer((256,), "uint16"),
    ue8x2_source: T.Buffer((256,), "uint16"),
    rbits_source: T.Buffer((256,), "uint32"),
    pack_f4: T.Buffer((4, 256), "uint8"),
    unpack_f4: T.Buffer((2, 256), "uint32"),
    stochastic_f8: T.Buffer((4, 256), "uint32"),
    stochastic_f4: T.Buffer((2, 256), "uint16"),
    exponent: T.Buffer((8, 256), "uint16"),
    from_exponent: T.Buffer((1, 256), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        index: T.let = step * 32 + lane
        second: T.let = (index + 1) % 256
        third: T.let = (index + 2) % 256
        fourth: T.let = (index + 3) % 256
        a: T.let = f32_source[index]
        b: T.let = f32_source[second]
        e: T.let = f32_source[third]
        f: T.let = f32_source[fourth]
        rbits: T.let = rbits_source[index]
        packed_f4: T.let = T.cast(f4x2_source[index], "uint8")

        # cvt.rn.satfinite{.relu}.e2m1x2.f32
        T.ptx["cvt.rn.satfinite.e2m1x2.f32"](pack_f4[0, index], a, b)
        T.ptx["cvt.rn.satfinite.relu.e2m1x2.f32"](pack_f4[1, index], a, b)

        # cvt.rn.satfinite{.relu}.e2m1x2.f16x2
        T.ptx["cvt.rn.satfinite.e2m1x2.f16x2"](pack_f4[2, index], f16x2_source[index])
        T.ptx["cvt.rn.satfinite.relu.e2m1x2.f16x2"](pack_f4[3, index], f16x2_source[index])

        # cvt.rn{.relu}.f16x2.e2m1x2
        T.ptx["cvt.rn.f16x2.e2m1x2"](unpack_f4[0, index], packed_f4)
        T.ptx["cvt.rn.relu.f16x2.e2m1x2"](unpack_f4[1, index], packed_f4)

        # cvt.rs{.relu}.satfinite.{e4m3x4,e5m2x4}.f32
        T.ptx["cvt.rs.satfinite.e4m3x4.f32"](stochastic_f8[0, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.relu.satfinite.e4m3x4.f32"](stochastic_f8[1, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.satfinite.e5m2x4.f32"](stochastic_f8[2, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.relu.satfinite.e5m2x4.f32"](stochastic_f8[3, index], a, b, e, f, rbits)

        # cvt.rs{.relu}.satfinite.e2m1x4.f32
        T.ptx["cvt.rs.satfinite.e2m1x4.f32"](stochastic_f4[0, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.relu.satfinite.e2m1x4.f32"](stochastic_f4[1, index], a, b, e, f, rbits)

        # cvt.{rz,rp}{.satfinite}.ue8m0x2.f32
        T.ptx["cvt.rz.ue8m0x2.f32"](exponent[0, index], a, b)
        T.ptx["cvt.rz.satfinite.ue8m0x2.f32"](exponent[1, index], a, b)
        T.ptx["cvt.rp.ue8m0x2.f32"](exponent[2, index], a, b)
        T.ptx["cvt.rp.satfinite.ue8m0x2.f32"](exponent[3, index], a, b)

        # cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2
        T.ptx["cvt.rz.ue8m0x2.bf16x2"](exponent[4, index], bf16x2_source[index])
        T.ptx["cvt.rz.satfinite.ue8m0x2.bf16x2"](exponent[5, index], bf16x2_source[index])
        T.ptx["cvt.rp.ue8m0x2.bf16x2"](exponent[6, index], bf16x2_source[index])
        T.ptx["cvt.rp.satfinite.ue8m0x2.bf16x2"](exponent[7, index], bf16x2_source[index])

        # cvt.rn.bf16x2.ue8m0x2
        T.ptx["cvt.rn.bf16x2.ue8m0x2"](from_exponent[0, index], ue8x2_source[index])


@T.prim_func
def narrow_cvt_bf16x2_forms(
    bf16x2_source: T.Buffer((256,), "uint32"),
    f4x2_source: T.Buffer((256,), "uint16"),
    f8x2_source: T.Buffer((256,), "uint16"),
    scale_source: T.Buffer((256,), "uint16"),
    pack_f4: T.Buffer((2, 256), "uint8"),
    unpack_f4: T.Buffer((4, 256), "uint32"),
    scaled: T.Buffer((12, 256), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        index: T.let = step * 32 + lane
        packed_f4: T.let = T.cast(f4x2_source[index], "uint8")
        packed_f8: T.let = f8x2_source[index]
        scale: T.let = scale_source[index]

        # cvt.rn.satfinite{.relu}.e2m1x2.bf16x2
        T.ptx["cvt.rn.satfinite.e2m1x2.bf16x2"](pack_f4[0, index], bf16x2_source[index])
        T.ptx["cvt.rn.satfinite.relu.e2m1x2.bf16x2"](pack_f4[1, index], bf16x2_source[index])

        # cvt.rn{.relu}{.satfinite}.bf16x2.e2m1x2
        T.ptx["cvt.rn.bf16x2.e2m1x2"](unpack_f4[0, index], packed_f4)
        T.ptx["cvt.rn.relu.bf16x2.e2m1x2"](unpack_f4[1, index], packed_f4)
        T.ptx["cvt.rn.satfinite.bf16x2.e2m1x2"](unpack_f4[2, index], packed_f4)
        T.ptx["cvt.rn.relu.satfinite.bf16x2.e2m1x2"](unpack_f4[3, index], packed_f4)

        # cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.e4m3x2
        T.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2"](scaled[0, index], packed_f8, scale)
        T.ptx["cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e4m3x2"](scaled[1, index], packed_f8, scale)
        T.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2"](
            scaled[2, index], packed_f8, scale
        )
        T.ptx["cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2"](
            scaled[3, index], packed_f8, scale
        )

        # cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.e5m2x2
        T.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.e5m2x2"](scaled[4, index], packed_f8, scale)
        T.ptx["cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e5m2x2"](scaled[5, index], packed_f8, scale)
        T.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e5m2x2"](
            scaled[6, index], packed_f8, scale
        )
        T.ptx["cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e5m2x2"](
            scaled[7, index], packed_f8, scale
        )

        # cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.e2m1x2
        T.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2"](scaled[8, index], packed_f4, scale)
        T.ptx["cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e2m1x2"](scaled[9, index], packed_f4, scale)
        T.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e2m1x2"](
            scaled[10, index], packed_f4, scale
        )
        T.ptx["cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e2m1x2"](
            scaled[11, index], packed_f4, scale
        )


# --- copied from tests/numsim/microtests/cases/ptx_cvt_scalar.py ---

@T.prim_func
def scalar_cvt_narrowing(
    source_f32: T.Buffer((16,), "float32"),
    out_f16: T.Buffer((16, 8), "uint16"),
    out_bf16: T.Buffer((16, 8), "uint16"),
    out_tf32: T.Buffer((16, 10), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.ptx["cvt.rn.f16.f32"](out_f16[lane, 0], source_f32[lane])
        T.ptx["cvt.rn.relu.f16.f32"](out_f16[lane, 1], source_f32[lane])
        T.ptx["cvt.rn.satfinite.f16.f32"](out_f16[lane, 2], source_f32[lane])
        T.ptx["cvt.rn.relu.satfinite.f16.f32"](out_f16[lane, 3], source_f32[lane])
        T.ptx["cvt.rz.f16.f32"](out_f16[lane, 4], source_f32[lane])
        T.ptx["cvt.rz.relu.f16.f32"](out_f16[lane, 5], source_f32[lane])
        T.ptx["cvt.rz.satfinite.f16.f32"](out_f16[lane, 6], source_f32[lane])
        T.ptx["cvt.rz.relu.satfinite.f16.f32"](out_f16[lane, 7], source_f32[lane])
        T.ptx["cvt.rn.bf16.f32"](out_bf16[lane, 0], source_f32[lane])
        T.ptx["cvt.rn.relu.bf16.f32"](out_bf16[lane, 1], source_f32[lane])
        T.ptx["cvt.rn.satfinite.bf16.f32"](out_bf16[lane, 2], source_f32[lane])
        T.ptx["cvt.rn.relu.satfinite.bf16.f32"](out_bf16[lane, 3], source_f32[lane])
        T.ptx["cvt.rz.bf16.f32"](out_bf16[lane, 4], source_f32[lane])
        T.ptx["cvt.rz.relu.bf16.f32"](out_bf16[lane, 5], source_f32[lane])
        T.ptx["cvt.rz.satfinite.bf16.f32"](out_bf16[lane, 6], source_f32[lane])
        T.ptx["cvt.rz.relu.satfinite.bf16.f32"](out_bf16[lane, 7], source_f32[lane])
        T.ptx["cvt.rn.tf32.f32"](out_tf32[lane, 0], source_f32[lane])
        T.ptx["cvt.rn.satfinite.tf32.f32"](out_tf32[lane, 1], source_f32[lane])
        T.ptx["cvt.rn.relu.tf32.f32"](out_tf32[lane, 2], source_f32[lane])
        T.ptx["cvt.rn.satfinite.relu.tf32.f32"](out_tf32[lane, 3], source_f32[lane])
        T.ptx["cvt.rz.tf32.f32"](out_tf32[lane, 4], source_f32[lane])
        T.ptx["cvt.rz.satfinite.tf32.f32"](out_tf32[lane, 5], source_f32[lane])
        T.ptx["cvt.rz.relu.tf32.f32"](out_tf32[lane, 6], source_f32[lane])
        T.ptx["cvt.rz.satfinite.relu.tf32.f32"](out_tf32[lane, 7], source_f32[lane])
        T.ptx["cvt.rna.tf32.f32"](out_tf32[lane, 8], source_f32[lane])
        T.ptx["cvt.rna.satfinite.tf32.f32"](out_tf32[lane, 9], source_f32[lane])


# --- copied from tests/numsim/runtime/test_approximate_f32_contract.py ---

@T.prim_func
def approximate_f32_calls(
    source: T.Buffer((32,), "float32"),
    exponentials: T.Buffer((32,), "float32"),
    reciprocals: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    exponential = T.local_scalar("float32")
    reciprocal = T.local_scalar("float32")
    T.ptx.ex2.approx.ftz.f32(exponential, source[lane])
    T.ptx.rcp.approx.ftz.f32(reciprocal, source[lane])
    exponentials[lane] = exponential
    reciprocals[lane] = reciprocal


@T.prim_func
def bf16x2_exp2_calls(
    source: T.Buffer((32,), "uint32"),
    exponentials: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    exponential = T.local_scalar("uint32")
    T.ptx["ex2.approx.ftz.bf16x2"](exponential, source[lane])
    exponentials[lane] = exponential


# --- copied from tests/numsim/runtime/test_dense_mma_forms.py ---

@T.prim_func
def ptx_mma_s8_u8_m16n8k32_no_c(
    a: T.Buffer((16, 32), "int8"), b: T.Buffer((32, 8), "uint8"), output: T.Buffer((16, 8), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((16,), "int8")
    b_regs = T.alloc_local((8,), "uint8")
    d_regs = T.alloc_local((4,), "int32")
    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 4 + (register // 2) * 16)
        for packed in T.unroll(4):
            a_regs[register * 4 + packed] = a[row, col + packed]
    for register in T.unroll(2):
        row = T.meta_var(thread * 4 + register * 16)
        for packed in T.unroll(4):
            b_regs[register * 4 + packed] = b[row + packed, group]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")
    d_words = d_regs.view("uint32")
    T.ptx.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32(
        *[d_words[index] for index in range(4)],
        *[a_words[index] for index in range(4)],
        *[b_words[index] for index in range(2)],
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f16_accumulator_m16n8k8(
    a: T.Buffer((16, 8), "float16"),
    b: T.Buffer((8, 8), "float16"),
    c: T.Buffer((16, 8), "float16"),
    output: T.Buffer((16, 8), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "float16")
    b_regs = T.alloc_local((2,), "float16")
    c_regs = T.alloc_local((4,), "float16")
    d_regs = T.alloc_local((4,), "float16")
    for index in T.unroll(2):
        a_regs[index] = a[group, thread * 2 + index]
        a_regs[index + 2] = a[group + 8, thread * 2 + index]
        b_regs[index] = b[thread * 2 + index, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")
    c_words = c_regs.view("uint32")
    d_words = d_regs.view("uint32")
    T.ptx.mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16(
        d_words[0],
        d_words[1],
        a_words[0],
        a_words[1],
        b_words[0],
        c_words[0],
        c_words[1],
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f64_m8n8k4(
    a: T.Buffer((8, 4), "float64"),
    b: T.Buffer((4, 8), "float64"),
    c: T.Buffer((8, 8), "float64"),
    output: T.Buffer((8, 8), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_reg = T.alloc_local((1,), "float64")
    b_reg = T.alloc_local((1,), "float64")
    c_regs = T.alloc_local((2,), "float64")
    d_regs = T.alloc_local((2,), "float64")
    a_reg[0] = a[group, thread]
    b_reg[0] = b[thread, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    T.ptx.mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64(
        d_regs[0], d_regs[1], a_reg[0], b_reg[0], c_regs[0], c_regs[1]
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]


# --- copied from tests/numsim/runtime/test_packed_f16x2_arithmetic.py ---

@T.prim_func
def scalar_f16_add(
    lhs: T.Buffer((32,), "uint16"),
    rhs: T.Buffer((32,), "uint16"),
    output: T.Buffer((32,), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.add.f16(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_f16x2_subtract(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.sub.f16x2(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_f16x2_multiply(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mul.f16x2(output[lane], lhs[lane], rhs[lane])


# --- copied from tests/numsim/runtime/test_ptx_compare_semantics.py ---

_FLOAT_COMPARISONS = (
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "equ",
    "neu",
    "ltu",
    "leu",
    "gtu",
    "geu",
    "num",
    "nan",
)


_BOOL_OPS = ("and", "or", "xor")


_CLASSES = ("finite", "infinite", "number", "notanumber", "normal", "subnormal")


def _comparison_kernel():
    parameters = (
        'lhs_f32: T.Buffer((32,), "float32")',
        'rhs_f32: T.Buffer((32,), "float32")',
        'lhs_f64: T.Buffer((32,), "float64")',
        'rhs_f64: T.Buffer((32,), "float64")',
        'lhs_f16: T.Buffer((32,), "uint16")',
        'rhs_f16: T.Buffer((32,), "uint16")',
        'lhs_bf16: T.Buffer((32,), "uint16")',
        'rhs_bf16: T.Buffer((32,), "uint16")',
        'lhs_f16x2: T.Buffer((32,), "uint32")',
        'rhs_f16x2: T.Buffer((32,), "uint32")',
        'lhs_bf16x2: T.Buffer((32,), "uint32")',
        'rhs_bf16x2: T.Buffer((32,), "uint32")',
        'lhs_i32: T.Buffer((32,), "int32")',
        'rhs_i32: T.Buffer((32,), "int32")',
        'lhs_i64: T.Buffer((32,), "int64")',
        'rhs_i64: T.Buffer((32,), "int64")',
        'lhs_u64: T.Buffer((32,), "uint64")',
        'rhs_u64: T.Buffer((32,), "uint64")',
        'predicate: T.Buffer((32,), "uint32")',
        'regular_set: T.Buffer((17, 32), "uint32")',
        'integer_set: T.Buffer((32,), "int32")',
        'float_set: T.Buffer((32,), "float32")',
        'double_set: T.Buffer((32,), "uint32")',
        'half_set: T.Buffer((17, 32), "uint32")',
        'half_scalar_set: T.Buffer((4, 32), "uint16")',
        'predicate_set: T.Buffer((12, 32), "uint32")',
        'classes: T.Buffer((12, 32), "uint32")',
    )
    lines = [
        "@T.prim_func",
        "def ptx_compare_semantics(",
        *(f"    {parameter}," for parameter in parameters),
        "):",
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
    ]
    for row, comparison in enumerate(_FLOAT_COMPARISONS):
        lines.append(
            f'    T.ptx["set.{comparison}.u32.f32"]('
            f"regular_set[{row}, lane], lhs_f32[lane], rhs_f32[lane])"
        )
        lines.append(
            f'    T.ptx["set.{comparison}.u32.f16x2"]('
            f"half_set[{row}, lane], lhs_f16x2[lane], rhs_f16x2[lane])"
        )
    for offset, bool_op in enumerate(_BOOL_OPS):
        lines.append(
            f'    T.ptx["set.eq.{bool_op}.u32.f32"]('
            f"regular_set[{14 + offset}, lane], lhs_f32[lane], rhs_f32[lane], "
            "T.ptx.pred(predicate[lane]))"
        )
        lines.append(
            f'    T.ptx["set.ne.{bool_op}.u32.f16x2"]('
            f"half_set[{14 + offset}, lane], lhs_f16x2[lane], rhs_f16x2[lane], "
            "T.ptx.pred(predicate[lane]))"
        )
    lines.extend(
        (
            '    T.ptx["set.lt.s32.s64"](integer_set[lane], lhs_i64[lane], rhs_i64[lane])',
            '    T.ptx["set.hi.f32.u64"](float_set[lane], lhs_u64[lane], rhs_u64[lane])',
            '    T.ptx["set.nan.u32.f64"](double_set[lane], lhs_f64[lane], rhs_f64[lane])',
            '    T.ptx["set.lt.f16.s32"](half_scalar_set[0, lane], lhs_i32[lane], rhs_i32[lane])',
            '    T.ptx["set.gt.u16.bf16"](half_scalar_set[1, lane], lhs_bf16[lane], rhs_bf16[lane])',
            '    T.ptx["set.eq.bf16.b16"](half_scalar_set[2, lane], lhs_f16[lane], rhs_f16[lane])',
            '    T.ptx["set.eq.ftz.f16.f16"](half_scalar_set[3, lane], lhs_f16[lane], rhs_f16[lane])',
            '    T.ptx["setp.ne.and.f32"](predicate_set[0, lane], lhs_f32[lane], rhs_f32[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.eq.or.f32"](predicate_set[1, lane], predicate_set[2, lane], lhs_f32[lane], rhs_f32[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.hi.u64"](predicate_set[3, lane], predicate_set[4, lane], lhs_u64[lane], rhs_u64[lane])',
            '    T.ptx["setp.ltu.and.ftz.f16"](predicate_set[5, lane], lhs_f16[lane], rhs_f16[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.ltu.ftz.f16x2"](predicate_set[6, lane], predicate_set[7, lane], lhs_f16x2[lane], rhs_f16x2[lane])',
            '    T.ptx["setp.geu.xor.bf16x2"](predicate_set[8, lane], predicate_set[9, lane], lhs_bf16x2[lane], rhs_bf16x2[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.nan.or.bf16"](predicate_set[10, lane], lhs_bf16[lane], rhs_bf16[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.ne.xor.s32"](predicate_set[11, lane], lhs_i32[lane], rhs_i32[lane], T.ptx.pred(predicate[lane]))',
        )
    )
    for offset, classification in enumerate(_CLASSES):
        lines.append(
            f'    T.ptx["testp.{classification}.f32"](classes[{offset}, lane], lhs_f32[lane])'
        )
        lines.append(
            f'    T.ptx["testp.{classification}.f64"](classes[{6 + offset}, lane], lhs_f64[lane])'
        )
    return tvm.script.from_source("\n".join(lines), {"T": T})


def _slct_kernel():
    return tvm.script.from_source(
        """
@T.prim_func
def ptx_slct_semantics(
    a16: T.Buffer((32,), "bfloat16"),
    b16: T.Buffer((32,), "int16"),
    c_i32: T.Buffer((32,), "int32"),
    a32: T.Buffer((32,), "float32"),
    b32: T.Buffer((32,), "int32"),
    c_f32: T.Buffer((32,), "float32"),
    a64: T.Buffer((32,), "uint64"),
    b64: T.Buffer((32,), "int64"),
    out16: T.Buffer((32,), "float16"),
    out32: T.Buffer((32,), "uint32"),
    out32_ftz: T.Buffer((32,), "int32"),
    out64: T.Buffer((32,), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["slct.b16.s32"](out16[lane], a16[lane], b16[lane], c_i32[lane])
    T.ptx["slct.f32.f32"](out32[lane], a32[lane], b32[lane], c_f32[lane])
    T.ptx["slct.ftz.f32.f32"](out32_ftz[lane], out32[lane], a32[lane], c_f32[lane])
    T.ptx["slct.b64.s32"](out64[lane], a64[lane], b64[lane], c_i32[lane])
""",
        {"T": T},
    )


ptx_compare_semantics = _comparison_kernel()


ptx_slct_semantics = _slct_kernel()


# --- copied from tests/numsim/runtime/test_ptx_new_register_semantics.py ---

@T.prim_func
def ptx_new_register_semantics(
    lhs_i32: T.Buffer((32,), "int32"),
    rhs_i32: T.Buffer((32,), "int32"),
    mul_i32: T.Buffer((32,), "int32"),
    sub_i32: T.Buffer((32,), "int32"),
    f16_lhs: T.Buffer((32,), "uint32"),
    f16_rhs: T.Buffer((32,), "uint32"),
    f16_addend: T.Buffer((32,), "uint32"),
    f16_out: T.Buffer((32,), "uint32"),
    bf16_lhs: T.Buffer((32,), "uint32"),
    bf16_rhs: T.Buffer((32,), "uint32"),
    bf16_addend: T.Buffer((32,), "uint32"),
    bf16_out: T.Buffer((32,), "uint32"),
    bit_lhs: T.Buffer((32,), "uint32"),
    bit_rhs: T.Buffer((32,), "uint32"),
    bit_or: T.Buffer((32,), "uint32"),
    prmt_a: T.Buffer((32,), "uint32"),
    prmt_b: T.Buffer((32,), "uint32"),
    prmt_selector: T.Buffer((32,), "uint32"),
    prmt_out: T.Buffer((32,), "uint32"),
    mul_hi_lhs: T.Buffer((32,), "uint32"),
    mul_hi_rhs: T.Buffer((32,), "uint32"),
    mul_hi_out: T.Buffer((32,), "uint32"),
    xor_lhs: T.Buffer((32,), "int32"),
    xor_rhs: T.Buffer((32,), "int32"),
    xor_out: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mul.lo.s32(mul_i32[lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx.sub.s32(sub_i32[lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx.fma.rn.f16x2(f16_out[lane], f16_lhs[lane], f16_rhs[lane], f16_addend[lane])
    T.ptx.fma.rn.bf16x2(bf16_out[lane], bf16_lhs[lane], bf16_rhs[lane], bf16_addend[lane])
    T.ptx.or_.b32(bit_or[lane], bit_lhs[lane], bit_rhs[lane])
    T.ptx.prmt.b32(prmt_out[lane], prmt_a[lane], prmt_b[lane], prmt_selector[lane])
    T.ptx.mul.hi.u32(mul_hi_out[lane], mul_hi_lhs[lane], mul_hi_rhs[lane])
    T.ptx.xor.b32(xor_out[lane], xor_lhs[lane], xor_rhs[lane])


# --- copied from tests/numsim/runtime/test_ptx_register_bits.py ---

@T.prim_func
def ptx_register_bits(
    input_u32: T.Buffer((32, 2), "uint32"),
    input_i32: T.Buffer((32, 2), "int32"),
    input_u64: T.Buffer((32, 2), "uint64"),
    input_f32: T.Buffer((32, 3), "float32"),
    output_u32: T.Buffer((32, 3), "uint32"),
    output_i32: T.Buffer((32,), "int32"),
    output_u64: T.Buffer((32, 3), "uint64"),
    output_f32: T.Buffer((32, 3), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed_u32 = T.alloc_local((1,), "uint64")
    packed_f32 = T.alloc_local((1,), "uint64")

    T.evaluate(T.ptx.mov.b64(packed_u32[0], input_u32[lane, 0], input_u32[lane, 1]))
    output_u64[lane, 0] = packed_u32[0]
    T.evaluate(T.ptx.mov.b64(output_u32[lane, 0], output_u32[lane, 1], packed_u32[0]))

    T.evaluate(T.ptx.mov.b64(packed_f32[0], input_f32[lane, 0], input_f32[lane, 1]))
    output_u64[lane, 1] = packed_f32[0]
    T.evaluate(T.ptx.mov.b64(output_f32[lane, 0], output_f32[lane, 1], packed_f32[0]))

    T.evaluate(T.ptx.add.u32(output_u32[lane, 2], input_u32[lane, 0], input_u32[lane, 1]))
    T.evaluate(T.ptx.add.s32(output_i32[lane], input_i32[lane, 0], input_i32[lane, 1]))
    T.evaluate(T.ptx.xor.b64(output_u64[lane, 2], input_u64[lane, 0], input_u64[lane, 1]))
    T.evaluate(T.ptx.abs.f32(output_f32[lane, 2], input_f32[lane, 2]))


@T.prim_func
def ptx_latest_canonical_scalar_forms(
    lhs_i32: T.Buffer((32,), "int32"),
    rhs_i32: T.Buffer((32,), "int32"),
    lhs_bf16: T.Buffer((32,), "uint16"),
    rhs_bf16: T.Buffer((32,), "uint16"),
    addend: T.Buffer((32,), "float32"),
    unary_input: T.Buffer((32,), "float32"),
    divisor: T.Buffer((32,), "float32"),
    maximum: T.Buffer((32,), "int32"),
    mixed_add: T.Buffer((32,), "float32"),
    mixed_sub: T.Buffer((32,), "float32"),
    mixed_fma: T.Buffer((32,), "float32"),
    negated: T.Buffer((32,), "float32"),
    quotient: T.Buffer((32,), "float32"),
    quotient_rn: T.Buffer((32,), "float32"),
    reciprocal: T.Buffer((32,), "float32"),
    inverse_sqrt: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["max.s32"](maximum[lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx["add.rn.f32.bf16"](mixed_add[lane], lhs_bf16[lane], addend[lane])
    T.ptx["sub.rn.f32.bf16"](mixed_sub[lane], lhs_bf16[lane], addend[lane])
    T.ptx["fma.rn.f32.bf16"](mixed_fma[lane], lhs_bf16[lane], rhs_bf16[lane], addend[lane])
    T.ptx["neg.ftz.f32"](negated[lane], unary_input[lane])
    T.ptx["div.approx.ftz.f32"](quotient[lane], unary_input[lane], divisor[lane])
    T.ptx["div.rn.f32"](quotient_rn[lane], unary_input[lane], divisor[lane])
    T.ptx["rcp.rn.f32"](reciprocal[lane], divisor[lane])
    T.ptx["rsqrt.approx.f32"](inverse_sqrt[lane], divisor[lane])


# --- copied from tests/numsim/runtime/test_ptx_warp_collectives.py ---

@T.prim_func
def raw_ptx_warp_collectives(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32, 6), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shuffled = T.alloc_local((1,), "uint32")
    shuffled_p = T.alloc_local((1,), "uint32")
    in_range = T.alloc_local((1,), "uint32")
    elected_lane = T.alloc_local((1,), "uint32")
    is_elected = T.alloc_local((1,), "uint32")
    full = T.uint32(0xFFFFFFFF)

    T.ptx.shfl_sync.idx.b32(
        shuffled[0],
        source[lane],
        T.cast((lane * 7 + 3) % 32, "uint32"),
        T.uint32(31),
        full,
    )
    output[lane, 0] = shuffled[0]
    T.ptx.shfl_sync.bfly.b32(shuffled[0], source[lane], T.uint32(1), T.uint32(31), full)
    output[lane, 1] = shuffled[0]
    T.ptx.shfl_sync.down.b32(
        shuffled_p[0],
        in_range[0],
        source[lane],
        T.uint32(2),
        T.uint32(31),
        full,
    )
    T.ptx.elect_sync(elected_lane[0], is_elected[0], full)
    output[lane, 2] = shuffled_p[0]
    output[lane, 3] = in_range[0]
    output[lane, 4] = elected_lane[0]
    output[lane, 5] = is_elected[0]


@T.prim_func
def raw_ptx_f32_reductions(
    source: T.Buffer((32,), "float32"),
    zeros: T.Buffer((32,), "float32"),
    output: T.Buffer((32, 6), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full = T.uint32(0xFFFFFFFF)
    T.ptx.redux_sync.max.NaN.f32(output[lane, 0], source[lane], full)
    T.ptx.redux_sync.min.NaN.f32(output[lane, 1], source[lane], full)
    T.ptx.redux_sync.max.f32(output[lane, 2], source[lane], full)
    T.ptx.redux_sync.min.f32(output[lane, 3], source[lane], full)
    T.ptx.redux_sync.max.f32(output[lane, 4], zeros[lane], full)
    T.ptx.redux_sync.min.f32(output[lane, 5], zeros[lane], full)


@T.prim_func
def raw_ptx_movmatrix_b16(
    source: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    transposed = T.alloc_local((1,), "uint32")
    T.ptx["movmatrix.sync.aligned.m8n8.trans.b16"](
        transposed[0],
        source[lane],
    )
    output[lane] = transposed[0]


# --- copied from tests/numsim/runtime/test_raw_tcgen_codegen.py ---

_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])


_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def raw_tcgen_ld_missing_shape_mappings(
    source: T.Buffer((128, 16), "uint32"), output: T.Buffer((3, 4, 32, 2), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    registers = T.alloc_local((2,), "uint32")
    for col in T.unroll(16):
        tmem[physical_row, col] = source[physical_row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x2.b32"](
        registers[0], registers[1], T.uint32(0), 2 * (2) if False else 2
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[0, warp, lane, 0] = registers[0]
    output[0, warp, lane, 1] = registers[1]
    T.ptx["tcgen05.ld.sync.aligned.16x64b.x2.b32"](registers[0], registers[1], T.uint32(0))
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[1, warp, lane, 0] = registers[0]
    output[1, warp, lane, 1] = registers[1]
    T.ptx["tcgen05.ld.sync.aligned.16x128b.x1.b32"](registers[0], registers[1], T.uint32(0))
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[2, warp, lane, 0] = registers[0]
    output[2, warp, lane, 1] = registers[1]


@T.prim_func
def raw_tcgen_mma_tf32_ts_predicated(
    a: T.Buffer((64, 8), "float32"),
    b_physical: T.Buffer((4096,), "uint8"),
    output: T.Buffer((64, 32), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64
    desc_b_low: T.uint32
    desc_b_replaced: T.uint64

    for copy_i in T.serial(128):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for k in T.unroll(8):
            tmem[physical_lane, k] = T.reinterpret("uint32", a[row, k])
        for col in T.serial(32):
            tmem[physical_lane, 16 + col] = T.reinterpret("uint32", T.float32(4))
    T.cuda.cta_sync()

    T.cuda.tcgen05.encode_instr_descriptor(
        T.address_of(desc_i),
        d_dtype="float32",
        a_dtype="tf32",
        b_dtype="tf32",
        M=64,
        N=32,
        K=8,
        trans_a=False,
        trans_b=False,
        n_cta_groups=1,
    )
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
    )
    desc_b_low = T.cast(desc_b, "uint32")
    desc_b_replaced = T.bitwise_or(
        T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0xFFFFFFFF))), T.cast(desc_b_low, "uint64")
    )
    T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
        T.uint32(16),
        T.uint32(0),
        desc_b_replaced,
        desc_i,
        T.uint32(1 << 3),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.ptx.pred(T.uint32(1)),
        pred=T.cast(lane == 7, "uint32"),
    )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.serial(32):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, 16 + col])


# --- copied from tests/numsim/runtime/test_scalar_control.py ---

@T.prim_func
def scalar_warp_intrinsics(
    output_u32: T.Buffer((32, 7), "uint32"),
    output_f32: T.Buffer((32, 4), "float32"),
    output_u64: T.Buffer((32,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full: T.let = T.uint32(0xFFFFFFFF)
    value: T.let = T.cast(lane + 1, "uint32")
    output_u32[lane, 0] = T.cuda.__shfl_up_sync(full, value, 1, 32)
    output_u32[lane, 1] = T.cuda.__shfl_down_sync(full, value, 1, 32)
    output_u32[lane, 2] = T.cuda.__shfl_xor_sync(full, value, 1, 32)
    output_u32[lane, 3] = T.cuda.__activemask()
    packed_float2: T.let = T.cuda.make_float2(T.cast(lane, "float32"), T.cast(lane + 1, "float32"))
    output_u32[lane, 4] = T.cuda.float22bfloat162_rn_from_float2(packed_float2)
    output_u32[lane, 5] = T.call_intrin(
        "uint32", "tirx.cuda.sm100_2sm_leader_smem_addr", T.uint64(0xFFFFFFFFF)
    )
    output_u32[lane, 6] = T.cuda.smem_addr_from_uint64(T.uint64(0x123456789))
    output_f32[lane, 0] = T.cuda.half2float(T.cast(T.cast(lane, "float32") + 0.25, "float16"))
    output_f32[lane, 1] = T.cuda.bfloat162float(T.cast(T.cast(lane, "float32") + 0.5, "bfloat16"))
    maximum = T.local_scalar("float32")
    minimum = T.local_scalar("float32")
    T.ptx.max.f32(maximum, T.cast(lane, "float32"), T.float32(17), T.float32(-3))
    T.ptx.min.f32(minimum, T.cast(lane, "float32"), T.float32(17), T.float32(-3))
    output_f32[lane, 2] = maximum
    output_f32[lane, 3] = minimum
    output_u64[lane] = T.cuda.clock64()


@T.prim_func
def current_scalar_device_intrinsics(
    output_f32: T.Buffer((3, 32), "float32"),
    output_u32: T.Buffer((3, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    T.evaluate(T.cuda.iket.mark("numsim"))
    sentinel: T.let = T.cuda.iket.sentinel_token("sentinel")
    T.evaluate(T.cuda.iket.range_end(sentinel))
    token: T.let = T.cuda.iket.range_start("range")
    T.evaluate(T.cuda.iket.range_push("stack"))
    T.evaluate(T.cuda.iket.range_pop())
    T.evaluate(T.cuda.iket.range_end(token))
    official: T.let = T.call_intrin("uint32", "tirx.cuda.iket_official_event", T.int32(7), "numsim")

    output_f32[0, lane] = T.cuda.fdividef(T.cast(lane + 1, "float32"), T.float32(2))
    T.ptx.cvt.rn.f32.s32(output_f32[1, lane], lane + 1)
    output_f32[2, lane] = T.log2(T.cast(lane + 1, "float32"))
    output_u32[0, lane] = sentinel
    output_u32[1, lane] = token
    output_u32[2, lane] = official


@T.prim_func
def packed_f16x2_conversion(
    high: T.Buffer((32,), "float32"),
    low: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.cvt.rn.f16x2.f32(output[lane], high[lane], low[lane])


@T.prim_func
def explicit_mapa_forms(output: T.Buffer((32,), "int32"), addresses: T.Buffer((32,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    shared[lane] = lane + 37
    T.cuda.cta_sync()
    generic_u64 = T.local_scalar("uint64")
    cluster_u32 = T.local_scalar("uint32")
    cluster_u64 = T.local_scalar("uint64")
    shared_address: T.let = T.cuda.cvta_generic_to_shared(T.address_of(shared[1]))
    T.ptx.mapa.u64(generic_u64, T.address_of(shared[1]), T.uint32(0))
    T.ptx.mapa.shared__cluster.u32(cluster_u32, shared_address, T.uint32(0))
    T.ptx.mapa.shared__cluster.u64(cluster_u64, T.address_of(shared[1]), T.uint32(0))
    mapped_ptr: T.let[
        T.Var(name="explicit_mapa_shared_cluster_u64", ty=PointerType(PrimType("int32"), "shared"))
    ] = T.reinterpret(
        PointerType(PrimType("int32"), "shared"),
        cluster_u64,
    )
    mapped = T.decl_buffer((1,), "int32", scope="shared", data=mapped_ptr)
    output[lane] = mapped[0]
    addresses[lane] = T.cuda.cvta_generic_to_shared(T.address_of(shared[1]))


@T.prim_func
def explicit_cvta_shared_cluster_u64(addresses: T.Buffer((32,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    T.ptx.cvta.to.shared__cluster.u64(addresses[lane], T.address_of(shared[lane]))


# --- copied from tests/numsim/runtime/test_scalar_helpers.py ---

@T.prim_func
def mega_scalar_helpers(
    packed_lhs: T.Buffer((32,), "uint32"),
    packed_rhs: T.Buffer((32,), "uint32"),
    fp8_values: T.Buffer((32, 4), "float32"),
    masks: T.Buffer((32,), "uint32"),
    bases: T.Buffer((32,), "uint32"),
    offsets: T.Buffer((32,), "int32"),
    accum: T.Buffer((32,), "float32"),
    bf16_values: T.Buffer((32,), "uint16"),
    packed_output: T.Buffer((32, 4), "uint32"),
    float_output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed_output[lane, 0] = T.cuda.hmin2(packed_lhs[lane], packed_rhs[lane])
    packed_output[lane, 1] = T.cuda.hmax2(packed_lhs[lane], packed_rhs[lane])
    packed_output[lane, 2] = T.cuda.fp8x4_e4m3_from_float4(
        fp8_values[lane, 0], fp8_values[lane, 1], fp8_values[lane, 2], fp8_values[lane, 3]
    )
    T.ptx.fns.b32(packed_output[lane, 3], masks[lane], bases[lane], offsets[lane])
    T.ptx.add.rn.f32.bf16(float_output[lane], bf16_values[lane], accum[lane])


# --- copied from tests/numsim/runtime/test_sparse_mma_forms.py ---

@T.prim_func
def sparse_tf32_m16n8k8(
    packed_a: T.Buffer((16, 4), "float32"),
    b: T.Buffer((8, 8), "float32"),
    c: T.Buffer((16, 8), "float32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((2,), "float32")
    b_regs = T.alloc_local((2,), "float32")
    acc = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")
    a_packed = a_regs.view("uint32")
    b_packed = b_regs.view("uint32")
    a_regs[0] = packed_a[group, thread]
    a_regs[1] = packed_a[group + 8, thread]
    b_regs[0] = b[thread, group]
    b_regs[1] = b[thread + 4, group]
    acc[0] = c[group, thread * 2]
    acc[1] = c[group, thread * 2 + 1]
    acc[2] = c[group + 8, thread * 2]
    acc[3] = c[group + 8, thread * 2 + 1]
    metadata[0] = metadata_words[lane]
    T.ptx["mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"](
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_packed[0],
        a_packed[1],
        b_packed[0],
        b_packed[1],
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        metadata[0],
        2,
    )
    output[group, thread * 2] = acc[0]
    output[group, thread * 2 + 1] = acc[1]
    output[group + 8, thread * 2] = acc[2]
    output[group + 8, thread * 2 + 1] = acc[3]


@T.prim_func
def sparse_s8_u8_m16n8k64(
    packed_a: T.Buffer((16, 32), "int8"),
    b: T.Buffer((64, 8), "uint8"),
    c: T.Buffer((16, 8), "int32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((16, 8), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((16,), "int8")
    b_regs = T.alloc_local((16,), "uint8")
    acc = T.alloc_local((4,), "int32")
    metadata = T.alloc_local((1,), "uint32")
    a_packed = a_regs.view("uint32")
    b_packed = b_regs.view("uint32")
    acc_packed = acc.view("uint32")
    for slot in T.unroll(16):
        half = T.meta_var(slot // 8)
        within = T.meta_var(slot % 8)
        row = T.meta_var(group + (within // 4) * 8)
        a_regs[slot] = packed_a[row, half * 16 + thread * 4 + within % 4]
        inner = T.meta_var(16 * (slot // 4) + 4 * thread + slot % 4)
        b_regs[slot] = b[inner, group]
    acc[0] = c[group, thread * 2]
    acc[1] = c[group, thread * 2 + 1]
    acc[2] = c[group + 8, thread * 2]
    acc[3] = c[group + 8, thread * 2 + 1]
    metadata[0] = metadata_words[lane]
    T.ptx["mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.u8.s32"](
        acc_packed[0],
        acc_packed[1],
        acc_packed[2],
        acc_packed[3],
        a_packed[0],
        a_packed[1],
        a_packed[2],
        a_packed[3],
        b_packed[0],
        b_packed[1],
        b_packed[2],
        b_packed[3],
        acc_packed[0],
        acc_packed[1],
        acc_packed[2],
        acc_packed[3],
        metadata[0],
        0,
    )
    output[group, thread * 2] = acc[0]
    output[group, thread * 2 + 1] = acc[1]
    output[group + 8, thread * 2] = acc[2]
    output[group + 8, thread * 2 + 1] = acc[3]


@T.prim_func
def sparse_float8_m16n8k64_zero(output: T.Buffer((32, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a_regs = T.alloc_local((16,), "float8_e4m3fn")
    b_regs = T.alloc_local((16,), "float8_e4m3fn")
    acc = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")
    a_packed = a_regs.view("uint32")
    b_packed = b_regs.view("uint32")
    for index in T.unroll(16):
        a_regs[index] = T.cast(T.float32(0), "float8_e4m3fn")
        b_regs[index] = T.cast(T.float32(0), "float8_e4m3fn")
    for index in T.unroll(4):
        acc[index] = T.float32(0)
    metadata[0] = T.uint32(0x44444444)
    T.ptx["mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32"](
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_packed[0],
        a_packed[1],
        a_packed[2],
        a_packed[3],
        b_packed[0],
        b_packed[1],
        b_packed[2],
        b_packed[3],
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output[lane, index] = acc[index]


# --- copied from tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py ---

@T.prim_func
def additional_scalar_payload_ops(output: T.Buffer((32, 3), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = T.cast(lane + 1, "float32")

    T.ptx.max.f32(output[lane, 0], value, T.float32(17))
    T.ptx.min.f32(output[lane, 1], value, T.float32(17))
    T.ptx.neg.f32(output[lane, 2], value)


@T.prim_func
def direct_lg2_payload(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.lg2.approx.ftz.f32(result[0], source[lane]))
    output[lane] = result[0]


@T.prim_func
def direct_tensor_map_payload_ops(
    source_map: T.TensorMap(),
    replacement: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        payload = T.decl_buffer(
            (8,),
            "uint64",
            data=T.reinterpret("handle", T.address_of(source_map)),
            scope="param",
            align=16,
        )
        T.ptx.st.global_.v4.b64(descriptor, payload[0], payload[1], payload[2], payload[3])
        T.ptx.st.global_.v4.b64(
            T.reinterpret("handle", T.reinterpret("uint64", descriptor) + T.uint64(32)),
            payload[4],
            payload[5],
            payload[6],
            payload[7],
        )
        T.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
            descriptor, T.reinterpret("uint64", replacement.data)
        )
        T.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(descriptor, 0, T.uint32(4))
        T.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(descriptor, 1, T.uint32(3))
        T.ptx.tensormap_replace.tile.global_stride.global_.b1024.b64(descriptor, 0, T.uint64(16))
        T.ptx.fence.proxy.tensormap__generic.release.gpu()
        T.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0, 0]), descriptor, 0, 0, T.address_of(barriers[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 48)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        for row in T.serial(3):
            for column in T.serial(4):
                output[row, column] = shared[row, column]


@T.prim_func
def mma_f16c_f32d_zero(output: T.Buffer((32, 8), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a_values = T.alloc_local((4,), "float16")
    b_values = T.alloc_local((4,), "float16")
    c_values = T.alloc_local((8,), "float16")
    d_values = T.alloc_local((8,), "float32")
    for index in T.unroll(4):
        a_values[index] = T.float16(0)
        b_values[index] = T.float16(0)
    for index in T.unroll(8):
        c_values[index] = T.float16(0)
    a_words = a_values.view("uint32")
    b_words = b_values.view("uint32")
    c_words = c_values.view("uint32")

    T.ptx.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16(
        *[d_values[index] for index in range(8)],
        a_words[0],
        a_words[1],
        b_words[0],
        b_words[1],
        *[c_words[index] for index in range(4)],
    )
    for index in T.unroll(8):
        output[lane, index] = d_values[index]


@T.prim_func
def additional_sparse_mma_payload_ops(
    output_f16: T.Buffer((2, 32, 4), "float16"),
    output_i32: T.Buffer((32, 4), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a_f16 = T.alloc_local((8,), "float16")
    b_f16 = T.alloc_local((8,), "float16")
    acc_f16 = T.alloc_local((4,), "float16")
    a_i8 = T.alloc_local((8,), "int8")
    b_u8 = T.alloc_local((8,), "uint8")
    acc_i32 = T.alloc_local((4,), "int32")
    metadata = T.alloc_local((1,), "uint32")
    for index in T.unroll(8):
        a_f16[index] = T.float16(0)
        b_f16[index] = T.float16(0)
        a_i8[index] = T.int8(0)
        b_u8[index] = T.uint8(0)
    for index in T.unroll(4):
        acc_f16[index] = T.float16(0)
        acc_i32[index] = T.int32(0)
    a_f16_words = a_f16.view("uint32")
    b_f16_words = b_f16.view("uint32")
    acc_f16_words = acc_f16.view("uint32")
    a_i8_words = a_i8.view("uint32")
    b_u8_words = b_u8.view("uint32")
    acc_i32_words = acc_i32.view("uint32")
    metadata[0] = T.uint32(0x44444444)

    T.ptx.mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16(
        acc_f16_words[0],
        acc_f16_words[1],
        a_f16_words[0],
        a_f16_words[1],
        b_f16_words[0],
        b_f16_words[1],
        acc_f16_words[0],
        acc_f16_words[1],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output_f16[0, lane, index] = acc_f16[index]

    T.ptx.mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16(
        acc_f16_words[0],
        acc_f16_words[1],
        *[a_f16_words[index] for index in range(4)],
        *[b_f16_words[index] for index in range(4)],
        acc_f16_words[0],
        acc_f16_words[1],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output_f16[1, lane, index] = acc_f16[index]

    T.ptx.mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32(
        *[acc_i32_words[index] for index in range(4)],
        a_i8_words[0],
        a_i8_words[1],
        b_u8_words[0],
        b_u8_words[1],
        *[acc_i32_words[index] for index in range(4)],
        metadata[0],
        0,
    )
    for index in T.unroll(4):
        output_i32[lane, index] = acc_i32[index]


@T.prim_func
def tcgen05_st_split_roundtrip(
    source: T.Buffer((4, 32, 2), "uint32"),
    output: T.Buffer((128, 16), "uint32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 16),
        "uint32",
        scope="tmem",
        layout=_TMEM_D_16,
        allocated_addr=0,
    )
    registers = T.alloc_local((2,), "uint32")
    for column in T.unroll(16):
        tmem[physical_row, column] = T.uint32(0)
    registers[0] = source[warp, lane, 0]
    registers[1] = source[warp, lane, 1]
    T.ptx["tcgen05.st.sync.aligned.16x32bx2.x2.b32"](
        T.uint32(0),
        2,
        registers[0],
        registers[1],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for column in T.unroll(16):
        output[physical_row, column] = tmem[physical_row, column]


_KERNELS = (
    pass_emitted_cp_async_raw,
    raw_cp_async_zero_fill,
    current_scalar_device_intrinsics,
    packed_f16x2_conversion,
    packed_ptx_cvt,
    packed_fp8_cvt_sm100_forms,
    narrow_cvt_bf16x2_forms,
    narrow_cvt_sm100_forms,
    scalar_cvt_narrowing,
    scalar_f16_add,
    packed_f16x2_multiply,
    packed_f16x2_subtract,
    ptx_register_bits,
    ptx_new_register_semantics,
    ptx_compare_semantics,
    ptx_slct_semantics,
    ptx_latest_canonical_scalar_forms,
    raw_ptx_movmatrix_b16,
    raw_ptx_warp_collectives,
    raw_ptx_f32_reductions,
    approximate_f32_calls,
    bf16x2_exp2_calls,
    direct_lg2_payload,
    mega_scalar_helpers,
    explicit_mapa_forms,
    explicit_cvta_shared_cluster_u64,
    scalar_warp_intrinsics,
    additional_scalar_payload_ops,
    ptx_mma_f16_accumulator_m16n8k8,
    mma_f16c_f32d_zero,
    ptx_mma_f64_m8n8k4,
    ptx_mma_s8_u8_m16n8k32_no_c,
    sparse_tf32_m16n8k8,
    sparse_float8_m16n8k64_zero,
    sparse_s8_u8_m16n8k64,
    additional_sparse_mma_payload_ops,
    tcgen_commit_runtime_multicast,
    raw_tcgen_ld_missing_shape_mappings,
    raw_tcgen_mma_tf32_ts_predicated,
    tcgen05_st_split_roundtrip,
    direct_tensor_map_payload_ops,
)


def _zero_argument(parameter: Any) -> Any:
    return np.zeros(
        tuple(int(extent) for extent in parameter.shape),
        dtype=np.dtype(str(parameter.dtype)),
    )


def _kernel_name(kernel: Any) -> str:
    """Return the global name for decorated and directly constructed PrimFuncs."""
    return getattr(kernel, "__name__", None) or str(kernel.attrs["global_symbol"])


_BY_NAME = {_kernel_name(kernel): kernel for kernel in _KERNELS}


def _arguments(kernel_name: str) -> dict[str, Any]:
    kernel = _BY_NAME[kernel_name]
    if kernel_name == "direct_tensor_map_payload_ops":
        source = np.zeros((3, 4), dtype=np.float32)
        return {
            "source_map": TensorMap(
                base=source,
                global_shape=(4, 3),
                global_strides=(16,),
                box_shape=(4, 3),
                element_strides=(1, 1),
            ).numpy(),
            "replacement": np.zeros((3, 4), dtype=np.float32),
            "output": np.zeros((3, 4), dtype=np.float32),
            "descriptor_storage": np.zeros(128, dtype=np.uint8),
        }
    arguments = {parameter.name: _zero_argument(parameter) for parameter in kernel.params}
    if kernel_name in ("sparse_tf32_m16n8k8", "sparse_s8_u8_m16n8k64"):
        arguments["metadata_words"] = np.full(32, np.uint32(0x44444444), dtype=np.uint32)
    return arguments


_GAPS = {
    "scalar_warp_intrinsics": v2_gap(
        "synccheck verdict incomplete: analysis_incomplete 'Unsupported: "
        "tirx.cuda.sm100_2sm_leader_smem_addr ... has no faithful pure-value model' "
        "(numsim's shared::cluster address encoding differs from the hardware bit 24)"
    ),
    "raw_tcgen_mma_tf32_ts_predicated": v2_gap(
        "synccheck verdict incomplete: analysis_incomplete 'Unsupported: not modeled: "
        "tmem[4352]: tmem lane 32 is outside warp 0's sub-partition' (direct TMEM "
        "buffer store from warp 0 to lanes 32..63; legacy accepted it)"
    ),
}


def _params():
    return [
        pytest.param(name, marks=(_GAPS[name],) if name in _GAPS else (), id=name)
        for name in _BY_NAME
    ]


@pytest.mark.parametrize("kernel_name", _params())
def test_payload_runtime(kernel_name):
    """Port of tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime.

    Dropped pins: stats["task_count"]/stats["completed_task_count"]
    and search["algorithm"] (see the module docstring).
    """

    report = v2.synccheck(
        _BY_NAME[kernel_name],
        _arguments(kernel_name),
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
