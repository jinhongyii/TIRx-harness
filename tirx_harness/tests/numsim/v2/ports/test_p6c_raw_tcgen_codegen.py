"""v2 ports of legacy ``tests/numsim/runtime/test_raw_tcgen_codegen.py`` gate tests.

Legacy rejected these kernels while generating Rust (``analyze`` +
``emit_rust_module``) or in ``analyze`` (``UnmodeledTIRxFormError``). The
copies drive ``v2.transpile`` and ``v2.Engine().run``; the generated-Rust
``MatrixDescriptorSm107`` text pin is dropped. Kernels copied verbatim.
"""

from __future__ import annotations

import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine

_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_k32_without_arch():
    """The architecture-neutral K-major B, N=16 CTA2 form shared by SM100 and SM107."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x10040490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_k32_extended_span_without_arch():
    """K=32 does not imply the SM100 address width; architecture does."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_arena = T.alloc_buffer((270336,), "uint8", scope="shared", align=1024)
    shared_a = T.decl_buffer((16384,), "uint8", data=shared_arena.data, scope="shared")
    shared_b = T.decl_buffer(
        (8192,), "uint8", data=shared_arena.data, elem_offset=262144, scope="shared"
    )
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        desc_b = T.bitwise_or(
            T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0x7FFF))),
            T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(shared_b.ptr_to([0])), T.uint32(4)),
                "uint64",
            ),
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x10210490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )


@T.prim_func
def shared_span_above_18_bits_without_sm107_descriptor():
    """A large ordinary/SM100 shared arena must not inherit Rubin's wider descriptor."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((262160,), "uint8", scope="shared")
    if lane == 0:
        shared[262159] = T.uint8(1)


@T.prim_func
def raw_tcgen_mma_f8f6f4_reserved_operand():
    """Operand format 2 belongs to TF32, not the narrow-float family."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32((2 << 7) | (2 << 10)),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta_group2_reserved_destination():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0x302104B0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta_group2_invalid_n():
    """MN-major B keeps the CTA2 N=32 granularity, so N=16 is invalid."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0x10050490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )


def _stop(error: v2.ExecutionError) -> dict:
    stops = [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops, error.diagnostics
    return stops[0]


def _run(kernel):
    return v2.Engine().run(v2.transpile(kernel), {})


def test_shared_span_above_18_bits_is_checked_where_a_descriptor_addresses_it():
    """Delta F5 of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_shared_span_above_18_bits_requires_a_supported_sm107_descriptor_form``.

    Legacy rejected any kernel whose shared window exceeded the 18-bit SM100
    descriptor address space, even one that builds no descriptor. v2 checks
    the bound where a descriptor addresses shared memory (the run stops with
    ``invalid_operand``), so this kernel, which only stores to its last byte
    and names no ``tirx.cuda_arch`` (no per-CTA capacity check), transpiles
    and runs.
    """

    result = _run(shared_span_above_18_bits_without_sm107_descriptor)
    assert result.status.get("kind") == "completed", result.status


def test_f8f6f4_cta2_k32_extended_shared_span_is_gated_by_arch_not_k_bit():
    """Port of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_f8f6f4_cta2_k32_extended_shared_span_is_gated_by_arch_not_k_bit``.

    sm_100a: legacy rejected the extended (above 18-bit) shared span while
    emitting Rust. v2 also rejects it at transpile, because the 270336-byte
    shared window is above the 227 KB per-CTA capacity of sm_100a (delta F5:
    such a CTA cannot launch). sm_107a: legacy emitted the
    ``MatrixDescriptorSm107`` variant (generated-Rust pin dropped); v2 runs
    the same kernel to completion.
    """

    kernel = raw_tcgen_mma_f8f6f4_cta2_k32_extended_span_without_arch

    with pytest.raises(UnsupportedTIRxError, match="270336 bytes of shared memory, above the 232448-byte"):
        v2.transpile(kernel.with_attr("tirx.cuda_arch", "sm_100a"))

    result = _run(kernel.with_attr("tirx.cuda_arch", "sm_107a"))
    assert result.status.get("kind") == "completed", result.status


@pytest.mark.parametrize(
    "arch",
    [
        pytest.param(None, id="missing"),
        pytest.param("sm_999a", id="unknown"),
    ],
)
def test_raw_tcgen_f8f6f4_cta2_requires_exact_kernel_architecture(arch):
    """Port of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_f8f6f4_cta2_requires_exact_kernel_architecture``.

    ``analyze`` + ``emit_rust_module`` become ``v2.transpile``; the rejection
    is kept as ``UnsupportedTIRxError`` (legacy message text not pinned).
    """

    kernel = raw_tcgen_mma_f8f6f4_cta2_k32_without_arch
    if arch is not None:
        kernel = kernel.with_attr("tirx.cuda_arch", arch)
    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(kernel)


@pytest.mark.parametrize(
    "kernel",
    [
        raw_tcgen_mma_f8f6f4_reserved_operand,
        raw_tcgen_mma_f8f6f4_cta_group2_reserved_destination,
        raw_tcgen_mma_f8f6f4_cta_group2_invalid_n,
    ],
    ids=(
        "f8f6f4_reserved_operand",
        "f8f6f4_cta2_f16",
        "f8f6f4_cta2_invalid_n",
    ),
)
def test_raw_tcgen_dense_mma_gate_names_the_exact_legal_set(kernel):
    """Port of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_dense_mma_gate_names_the_exact_legal_set``.

    Legacy rejected each unmodeled form in ``analyze``
    (``UnmodeledTIRxFormError`` on ``call:tirx.ptx.tcgen05_mma_ss``). v2
    accepts the kernel at transpile and validates the instruction descriptor
    when the mma executes: the run fails closed with an ``invalid_operand``
    error (stage moved from transpile to run; no delta row). Dropped: the
    ``target_id`` and message text. The kernels carry ``tirx.cuda_arch =
    sm_100a``: a CTA-pair f8f6f4 mma needs an exact architecture (W12, see
    ``test_raw_tcgen_f8f6f4_cta2_requires_exact_kernel_architecture``), as in
    legacy, so the descriptor gate is reached.
    """

    with pytest.raises(v2.ExecutionError) as caught:
        _run(kernel.with_attr("tirx.cuda_arch", "sm_100a"))
    stop = _stop(caught.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "invalid_operand", stop
