"""v2 port of the legacy matrix-family fail-closed acceptance test: legacy
``analyze(kernel).unsupported == ()`` becomes "``v2.transpile`` lowers the
kernel without ``UnsupportedTIRxError``". Kernels copied verbatim."""

from __future__ import annotations

from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def ptx_mma_tf32_m16n8k8(
    a: T.Buffer((16, 8), "float32"),
    b: T.Buffer((8, 8), "float32"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "float32")
    b_regs = T.alloc_local((2,), "float32")
    c_regs = T.alloc_local((4,), "float32")
    d_regs = T.alloc_local((4,), "float32")

    a_regs[0] = a[group, thread]
    a_regs[1] = a[group + 8, thread]
    a_regs[2] = a[group, thread + 4]
    a_regs[3] = a[group + 8, thread + 4]
    b_regs[0] = b[thread, group]
    b_regs[1] = b[thread + 4, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")

    T.ptx.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32(
        d_regs[0],
        d_regs[1],
        d_regs[2],
        d_regs[3],
        a_words[0],
        a_words[1],
        a_words[2],
        a_words[3],
        b_words[0],
        b_words[1],
        c_regs[0],
        c_regs[1],
        c_regs[2],
        c_regs[3],
    )

    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_sp_f16_m16n8k16(
    packed_a: T.Buffer((16, 8), "float16"),
    b: T.Buffer((16, 8), "float16"),
    c: T.Buffer((16, 8), "float32"),
    metadata_words: T.Buffer((32,), "uint32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "float16")
    b_regs = T.alloc_local((4,), "float16")
    acc = T.alloc_local((4,), "float32")
    metadata = T.alloc_local((1,), "uint32")

    a_regs[0] = packed_a[group, thread * 2]
    a_regs[1] = packed_a[group, thread * 2 + 1]
    a_regs[2] = packed_a[group + 8, thread * 2]
    a_regs[3] = packed_a[group + 8, thread * 2 + 1]
    b_regs[0] = b[thread * 2, group]
    b_regs[1] = b[thread * 2 + 1, group]
    b_regs[2] = b[thread * 2 + 8, group]
    b_regs[3] = b[thread * 2 + 9, group]
    acc[0] = c[group, thread * 2]
    acc[1] = c[group, thread * 2 + 1]
    acc[2] = c[group + 8, thread * 2]
    acc[3] = c[group + 8, thread * 2 + 1]
    metadata[0] = metadata_words[lane]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")

    T.ptx.mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_words[0],
        a_words[1],
        b_words[0],
        b_words[1],
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        metadata[0],
        1,
    )

    output[group, thread * 2] = acc[0]
    output[group, thread * 2 + 1] = acc[1]
    output[group + 8, thread * 2] = acc[2]
    output[group + 8, thread * 2 + 1] = acc[3]



def test_matrix_family_supports_dense_and_sparse_ptx():
    """Port of ``tests/numsim/runtime/test_matrix_instruction_codegen.py::test_matrix_family_supports_dense_and_sparse_ptx``.

    Legacy ``analyze(...).unsupported == ()`` -> ``v2.transpile`` succeeds
    and yields one kernel per PrimFunc.
    """

    for kernel in (ptx_mma_tf32_m16n8k8, ptx_mma_sp_f16_m16n8k16):
        module = v2.transpile(kernel)
        assert len(module.spec.kernels) == 1
