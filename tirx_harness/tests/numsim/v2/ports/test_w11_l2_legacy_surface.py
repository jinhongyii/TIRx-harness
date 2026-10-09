"""v2 copies of seven legacy tests that fail under v2 only because they use the
deprecated ``tirx.ptx_legacy.*`` spelling (numsim-behaviour-deltas L2: v2 rejects it
at transpile with ``UnsupportedTIRxError: builtin tirx.ptx_legacy.<op>``).

Each legacy function gets an expected-error test on its original kernel. Where the
same operation has a ``T.ptx.*`` table form that v2 lowers, a corrected kernel
asserts the legacy test's observable outputs:

- ``ldmatrix`` x2/x4 b16 (transposed and not): ``ldmatrix.sync.aligned.m8n8.{x2,x4}[.trans].shared.b16``
  with each lane passing its own row address. This is the per-lane row pointer the
  legacy ``shared_offset`` expressed.
- ``mma`` m16n8k16 f16xf16+f32: ``T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``
  over the same register fragments.

The two 8-bit transposed ``ldmatrix`` tests have no table counterpart. The table has
``.b16`` only, and legacy modelled the 8-bit transpose with its own gather fallback.
They are expected-error only. ``test_reused_physical_pointers_build_one_native_artifact``
pinned a legacy build oracle (rustc borrow checking of reused pointer bindings), which
is internal. Its ``reused_legacy_mma_pointer_bindings`` member keeps the
expected-error test, and the reused-binding numerics are covered by the corrected
``reused_mma_pointer_bindings`` kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


def _rejects_ptx_legacy(kernel) -> None:
    """numsim-behaviour-deltas L2: the deprecated ``tirx.ptx_legacy`` surface fails closed."""
    with pytest.raises(UnsupportedTIRxError, match="ptx_legacy"):
        v2.transpile(kernel)


# -- legacy kernels (verbatim) -------------------------------------------------


@T.prim_func
def legacy_ldmatrix_x2_trans(output: T.Buffer((128,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint16", scope="shared")
    local = T.alloc_buffer((4,), "uint16", scope="local")
    for element in T.serial(8):
        shared[lane * 8 + element] = T.Cast("uint16", lane * 8 + element)
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(True, 2, ".b16", local.data, 0, shared.data, lane * 8, dtype="uint16")
    )
    for element in T.serial(4):
        output[lane * 4 + element] = local[element]


@T.prim_func
def legacy_ldmatrix_i8_transpose_fallback(output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1024,), "uint8", scope="shared")
    local = T.alloc_buffer((16,), "uint8", scope="local")
    for element in T.serial(32):
        shared[lane * 32 + element] = T.Cast("uint8", lane * 32 + element)
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(True, 4, ".b16", local.data, 0, shared.data, 32, dtype="uint8")
    )
    for element in T.serial(16):
        output[lane * 16 + element] = local[element]


@T.prim_func
def legacy_ldmatrix_i8_transpose_two_warps(output: T.Buffer((1024,), "uint8")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread = warp * 32 + lane
    shared = T.alloc_buffer((1024,), "uint8", scope="shared")
    local = T.alloc_buffer((16,), "uint8", scope="local")
    for element in T.serial(16):
        shared[thread * 16 + element] = T.Cast("uint8", thread * 16 + element)
    T.cuda.cta_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(True, 4, ".b16", local.data, 0, shared.data, 32, dtype="uint8")
    )
    for element in T.serial(16):
        output[thread * 16 + element] = local[element]


@T.prim_func
def legacy_ldmatrix_i8_x4(output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    local = T.alloc_buffer((16,), "uint8", scope="local")
    for element in T.serial(16):
        shared[lane * 16 + element] = T.Cast("uint8", lane * 16 + element)
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(
            False, 4, ".b16", local.data, 0, shared.data, lane * 16, dtype="uint8"
        )
    )
    for element in T.serial(16):
        output[lane * 16 + element] = local[element]


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


@T.prim_func
def reused_legacy_mma_pointer_bindings(
    a_words: T.Buffer((32, 4), "uint32"),
    b_words: T.Buffer((32, 2), "uint32"),
    c_values: T.Buffer((32, 4), "float32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((2,), "uint32")
    c = T.alloc_local((4,), "float32")
    d = T.alloc_local((4,), "float32")
    for register in T.unroll(4):
        a[register] = a_words[lane, register]
        c[register] = c_values[lane, register]
    for register in T.unroll(2):
        b[register] = b_words[lane, register]
    a_pointer = a.ptr_to([0])
    b_pointer = b.ptr_to([0])
    accumulator_pointer = d.ptr_to([0])

    # The legacy form uses one accumulator pointer for both C and D.  All
    # three pointer bindings are reused verbatim by the second call.
    d[0] = c[0]
    d[1] = c[0]
    d[2] = c[2]
    d[3] = c[3]
    T.ptx_legacy.mma(
        "m16n8k16",
        "row",
        "col",
        "float16",
        "float16",
        "float32",
        a_pointer,
        0,
        b_pointer,
        0,
        accumulator_pointer,
        0,
        False,
        dtype="float32",
    )
    d[0] = c[0]
    d[1] = c[0]
    d[2] = c[2]
    d[3] = c[3]
    T.ptx_legacy.mma(
        "m16n8k16",
        "row",
        "col",
        "float16",
        "float16",
        "float32",
        a_pointer,
        0,
        b_pointer,
        0,
        accumulator_pointer,
        0,
        False,
        dtype="float32",
    )
    for register in T.unroll(4):
        output[lane, register] = T.reinterpret("uint32", d[register])


# -- corrected kernels: the same operations through the T.ptx table ----------


@T.prim_func
def table_ldmatrix_x2_trans(output: T.Buffer((128,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint16", scope="shared")
    words = T.alloc_local((2,), "uint32")
    for element in T.serial(8):
        shared[lane * 8 + element] = T.Cast("uint16", lane * 8 + element)
    T.cuda.warp_sync()
    # Lanes 0..15 name the 16 row addresses of the two 8x8 matrices.
    T.ptx["ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"](
        words[0], words[1], shared.ptr_to([(lane % 16) * 8])
    )
    halves = words.view("uint16")
    for element in T.serial(4):
        output[lane * 4 + element] = halves[element]


@T.prim_func
def table_ldmatrix_i8_x4(output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    words = T.alloc_local((4,), "uint32")
    for element in T.serial(16):
        shared[lane * 16 + element] = T.Cast("uint8", lane * 16 + element)
    T.cuda.warp_sync()
    T.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
        words[0], words[1], words[2], words[3], shared.ptr_to([lane * 16])
    )
    local = words.view("uint8")
    for element in T.serial(16):
        output[lane * 16 + element] = local[element]


@T.prim_func
def table_mma_f16_m16n8k16(
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
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")

    T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
        accumulator[0],
        accumulator[1],
        accumulator[2],
        accumulator[3],
        a_words[0],
        a_words[1],
        a_words[2],
        a_words[3],
        b_words[0],
        b_words[1],
        accumulator[0],
        accumulator[1],
        accumulator[2],
        accumulator[3],
    )

    output[group, thread * 2] = accumulator[0]
    output[group, thread * 2 + 1] = accumulator[1]
    output[group + 8, thread * 2] = accumulator[2]
    output[group + 8, thread * 2 + 1] = accumulator[3]


@T.prim_func
def reused_mma_pointer_bindings(
    a_words: T.Buffer((32, 4), "uint32"),
    b_words: T.Buffer((32, 2), "uint32"),
    c_values: T.Buffer((32, 4), "float32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((2,), "uint32")
    c = T.alloc_local((4,), "float32")
    d = T.alloc_local((4,), "float32")
    for register in T.unroll(4):
        a[register] = a_words[lane, register]
        c[register] = c_values[lane, register]
    for register in T.unroll(2):
        b[register] = b_words[lane, register]
    for _call in T.unroll(2):
        # The same A/B/D registers feed both calls; D doubles as C.
        d[0] = c[0]
        d[1] = c[0]
        d[2] = c[2]
        d[3] = c[3]
        T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
            d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1], d[0], d[1], d[2], d[3]
        )
    for register in T.unroll(4):
        output[lane, register] = T.reinterpret("uint32", d[register])


# -- tests ---------------------------------------------------------------------


def _x2_trans_expected() -> np.ndarray:
    expected = np.empty((32, 4), dtype=np.uint16)
    for matrix in range(2):
        for lane in range(32):
            row = lane // 4
            pair = lane % 4
            low = (matrix * 64) + (pair * 2) * 8 + row
            expected[lane, matrix * 2] = np.uint16(low)
            expected[lane, matrix * 2 + 1] = np.uint16(low + 8)
    return expected.reshape(-1)


def test_legacy_ldmatrix_transpose_preserves_b16_fragment_abi():
    """Copy of ``tests/numsim/runtime/test_memory_ops.py::test_legacy_ldmatrix_transpose_preserves_b16_fragment_abi``:
    the table ``ldmatrix.x2.trans.b16`` gives the legacy fragment layout exactly."""
    result = v2.Engine().run(v2.transpile(table_ldmatrix_x2_trans), {"output": np.zeros(128, dtype=np.uint16)})
    np.testing.assert_array_equal(result.outputs["output"], _x2_trans_expected())


def test_legacy_ldmatrix_transpose_preserves_b16_fragment_abi_legacy_spelling_rejected():
    """Original kernel: numsim-behaviour-deltas L2 (``tirx.ptx_legacy.ldmatrix`` rejected)."""
    _rejects_ptx_legacy(legacy_ldmatrix_x2_trans)


def test_legacy_ldmatrix_8bit_nontranspose_keeps_b16_fragment_abi():
    """Copy of ``tests/numsim/runtime/test_memory_ops.py::test_legacy_ldmatrix_8bit_nontranspose_keeps_b16_fragment_abi``:
    ``ldmatrix.x4.b16`` over 8-bit data distributes each lane's 4 bytes by the b16 ABI."""
    result = v2.Engine().run(v2.transpile(table_ldmatrix_i8_x4), {"output": np.zeros(512, dtype=np.uint8)})
    expected = np.empty((32, 16), dtype=np.uint8)
    for lane in range(32):
        row = lane // 4
        fragment = lane % 4
        for matrix in range(4):
            source = (matrix * 8 + row) * 16 + fragment * 4
            expected[lane, matrix * 4 : matrix * 4 + 4] = np.arange(source, source + 4, dtype=np.uint16).astype(
                np.uint8
            )
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_legacy_ldmatrix_8bit_nontranspose_keeps_b16_fragment_abi_legacy_spelling_rejected():
    """Original kernel: numsim-behaviour-deltas L2."""
    _rejects_ptx_legacy(legacy_ldmatrix_i8_x4)


def test_legacy_ldmatrix_8bit_transpose_matches_tirx_manual_gather():
    """``tests/numsim/runtime/test_memory_ops.py::test_legacy_ldmatrix_8bit_transpose_matches_tirx_manual_gather``,
    expected-error only: numsim-behaviour-deltas L2. The ``ldmatrix`` table has no
    8-bit transpose (``.b16`` only), and legacy modelled it with its own gather
    fallback, so no table kernel expresses the same operation."""
    _rejects_ptx_legacy(legacy_ldmatrix_i8_transpose_fallback)


def test_legacy_ldmatrix_8bit_transpose_uses_full_thread_index_across_warps():
    """``tests/numsim/runtime/test_memory_ops.py::test_legacy_ldmatrix_8bit_transpose_uses_full_thread_index_across_warps``,
    expected-error only, as above (numsim-behaviour-deltas L2)."""
    _rejects_ptx_legacy(legacy_ldmatrix_i8_transpose_two_warps)


def test_ptx_mma_legacy_executes_actual_pointer_offset_abi():
    """Copy of ``tests/numsim/runtime/test_matrix_instruction_codegen.py::test_ptx_mma_legacy_executes_actual_pointer_offset_abi``:
    the same fragments through the table ``mma.sync.m16n8k16`` give ``a @ b + c``."""
    a = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) % 5) / np.float16(2)
    b = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 7) / np.float16(4)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 32
    result = v2.Engine().run(
        v2.transpile(table_mma_f16_m16n8k16),
        {"a": a, "b": b, "c": c, "output": np.zeros((16, 8), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], a.astype(np.float32) @ b.astype(np.float32) + c)


def test_ptx_mma_legacy_executes_actual_pointer_offset_abi_legacy_spelling_rejected():
    """Original kernel: numsim-behaviour-deltas L2 (``tirx.ptx_legacy.mma`` rejected)."""
    _rejects_ptx_legacy(ptx_mma_legacy_f16_m16n8k16)


def _reused_inputs() -> dict:
    packed_fp16_ones = np.uint32(0x3C003C00)
    return {
        "a_words": np.full((32, 4), packed_fp16_ones, dtype=np.uint32),
        "b_words": np.full((32, 2), packed_fp16_ones, dtype=np.uint32),
        "c_values": np.arange(32 * 4, dtype=np.float32).reshape(32, 4) / 4,
        "output": np.zeros((32, 4), dtype=np.uint32),
    }


def test_reused_legacy_mma_reuses_a_b_and_accumulator_bindings_across_calls():
    """Copy of ``tests/numsim/runtime/test_physical_pointer_ownership.py::test_reused_legacy_mma_reuses_a_b_and_accumulator_bindings_across_calls``:
    reusing the same A/B/D registers for two table ``mma`` calls gives the legacy result."""
    inputs = _reused_inputs()
    result = v2.Engine().run(v2.transpile(reused_mma_pointer_bindings), dict(inputs))
    # Every A/B half is 1.0, so each product fragment is exactly 16.0 and d = 16 + c;
    # the second accumulator register is seeded from c[0].
    accumulators = inputs["c_values"].copy()
    accumulators[:, 1] = inputs["c_values"][:, 0]
    expected = (np.float32(16.0) + accumulators).view(np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_reused_legacy_mma_reuses_a_b_and_accumulator_bindings_across_calls_legacy_spelling_rejected():
    """Original kernel: numsim-behaviour-deltas L2."""
    _rejects_ptx_legacy(reused_legacy_mma_pointer_bindings)


def test_reused_physical_pointers_build_one_native_artifact():
    """``tests/numsim/runtime/test_physical_pointer_ownership.py::test_reused_physical_pointers_build_one_native_artifact``.

    The legacy oracle was rustc borrow checking of the generated artifact, which is a
    legacy internal with no v2 counterpart. The only member of the module that v2
    cannot transpile is ``reused_legacy_mma_pointer_bindings``
    (numsim-behaviour-deltas L2). Its reused-binding numerics are asserted by
    :func:`test_reused_legacy_mma_reuses_a_b_and_accumulator_bindings_across_calls`."""
    _rejects_ptx_legacy(reused_legacy_mma_pointer_bindings)
