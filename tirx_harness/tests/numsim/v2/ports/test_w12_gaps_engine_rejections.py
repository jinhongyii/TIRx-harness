"""v2 copies of W12-gaps 7-9: kernels legacy rejected at run time that v2
accepted until the engine grew the matching rule (W2).

Each copy runs the legacy kernel (copied verbatim) and asserts the v2 error
kind plus the site (the source line of the stopping diagnostic), never the
human text.

- Gap 7, ``test_memory_artifact.py::test_mega_mbarrier_init_rejects_partial_lane_aliasing``:
  ``mbarrier.init`` whose active lanes name neither one barrier nor one
  barrier per lane is a ``divergence`` error.
- Gap 8, ``test_cache_hint_ops.py::test_scalar_cache_hint_runtime_contracts_reach_the_engine``:
  the four scalar cache-hint range rules (applypriority 128-byte alignment,
  bulk prefetch size multiple of 16 and 16-byte alignment, non-tensor
  ``applypriority.async.bulk`` 128-byte alignment) are ``invalid_operand``
  errors at the faulting instruction.
- Gap 9, ``test_tcgen_collectors.py::test_typed_gemm_discards_a_raw_mma_fill``:
  a typed ``gemm_async`` between a raw ``collector::a::fill`` and
  ``collector::a::use`` discards the collector (missing ``.collector_usage``
  is ``::discard``), so the ``use`` is an ``invalid_operand`` error.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import tmem_datapath_layout

from tirx_harness.numsim import v2
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tests.numsim.v2.ports.test_error_kinds import _assert_stop

pytestmark = requires_v2_engine


def _run(kernel, inputs):
    return v2.Engine().run(v2.transpile(kernel), inputs)


# -- gap 7: tests/numsim/integration/test_memory_artifact.py ----------------


@T.prim_func
def invalid_partially_aliased_mbarrier_init(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((16,), "uint64", scope="shared")
    T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane // 2]), 1)
    if lane == 0:
        output[0] = 17


def test_mega_mbarrier_init_rejects_partial_lane_aliasing():
    with pytest.raises(v2.ExecutionError) as excinfo:
        _run(invalid_partially_aliased_mbarrier_init, {"output": np.zeros(1, dtype=np.int32)})
    _assert_stop(excinfo, ("divergence",), anchor="T.ptx.mbarrier.init.shared.b64(", warp=0)


# -- gap 8: tests/numsim/runtime/test_cache_hint_ops.py ---------------------


@T.prim_func
def invalid_scalar_cache_hint_contracts(
    source: T.Buffer((256,), "uint8"),
    apply_offset: T.int32,
    prefetch_offset: T.int32,
    prefetch_size: T.uint32,
    bulk_apply_offset: T.int32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = lane == 0
    T.ptx["applypriority.L2::evict_normal"](source.ptr_to([apply_offset]), pred=issue)
    T.ptx["cp.async.bulk.prefetch.L2.global.L2::evict_last"](
        source.ptr_to([prefetch_offset]), prefetch_size, pred=issue
    )
    T.ptx["applypriority.async.bulk.bulk_group.L2::evict_normal"](
        source.ptr_to([bulk_apply_offset]), T.uint32(16), pred=issue
    )


_VALID = {"apply_offset": 0, "prefetch_offset": 0, "prefetch_size": 16, "bulk_apply_offset": 0}


@pytest.mark.parametrize(
    ("override", "anchor"),
    [
        pytest.param(
            {"apply_offset": 16},
            'T.ptx["applypriority.L2::evict_normal"]',
            id="applypriority_align128",
        ),
        pytest.param(
            {"prefetch_size": 12}, 'T.ptx["cp.async.bulk.prefetch.L2', id="prefetch_size_multiple16"
        ),
        pytest.param(
            {"bulk_apply_offset": 16},
            'T.ptx["applypriority.async.bulk.bulk_group',
            id="bulk_applypriority_align128",
        ),
        pytest.param(
            {"prefetch_offset": 1}, 'T.ptx["cp.async.bulk.prefetch.L2', id="prefetch_align16"
        ),
    ],
)
def test_scalar_cache_hint_runtime_contracts_reach_the_engine(override, anchor):
    inputs = {"source": np.zeros(256, dtype=np.uint8), **_VALID, **override}
    with pytest.raises(v2.ExecutionError) as excinfo:
        _run(invalid_scalar_cache_hint_contracts, inputs)
    _assert_stop(excinfo, ("invalid_operand",), anchor=anchor, lanes=0x1, warp=0)


def test_scalar_cache_hint_contracts_accept_valid_operands():
    """Positive control: the same kernel with in-range operands runs clean."""
    result = _run(
        invalid_scalar_cache_hint_contracts, {"source": np.zeros(256, dtype=np.uint8), **_VALID}
    )
    assert result.status == {"kind": "completed"}, result.status


# -- gap 9: tests/numsim/runtime/test_tcgen_collectors.py -------------------


def _typed_gemm_between_fill_and_use(between_gemm: bool):
    a_layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_32B_ATOM, (128, 16))
    b_layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_32B_ATOM, (16, 16))
    d_layout = tmem_datapath_layout("D", 128, 16)

    @T.prim_func
    def kernel(out: T.Buffer((128, 16), "float32")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        a = T.alloc_shared((128, 16), "float16", layout=a_layout, align=512)
        b = T.alloc_shared((16, 16), "float16", layout=b_layout, align=512)
        d = T.decl_buffer((128, 16), "float32", scope="tmem", layout=d_layout, allocated_addr=0)
        desc_a: T.uint64
        desc_b: T.uint64
        if lane == 0:
            for row, col in T.grid(128, 16):
                a[row, col] = T.float16(1)
            for row, col in T.grid(16, 16):
                b[row, col] = T.float16(1)
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), a.ptr_to([0, 0]), ldo=0, sdo=16, swizzle=1
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), b.ptr_to([0, 0]), ldo=0, sdo=16, swizzle=1
            )
            T.ptx["tcgen05.mma.cta_group::1.kind::f16.collector::a::fill"](
                T.uint32(0),
                desc_a,
                desc_b,
                T.uint32((8 << 24) | (2 << 17) | 16),
                0,
                0,
                0,
                0,
                T.ptx.pred(T.uint32(0)),
            )
            if between_gemm:
                Tx.gemm_async(
                    d[:, :], a[:, :], b[:, :], accum=False, dispatch="tcgen05", cta_group=1
                )
            T.ptx["tcgen05.mma.cta_group::1.kind::f16.collector::a::use"](
                T.uint32(0),
                desc_a,
                desc_b,
                T.uint32((8 << 24) | (2 << 17) | 16),
                0,
                0,
                0,
                0,
                T.ptx.pred(T.uint32(1)),
            )
            out[0, 0] = d[0, 0]

    return kernel


def test_typed_gemm_discards_a_raw_mma_fill():
    with pytest.raises(v2.ExecutionError) as excinfo:
        _run(_typed_gemm_between_fill_and_use(True), {"out": np.zeros((128, 16), np.float32)})
    _assert_stop(
        excinfo,
        ("invalid_operand",),
        anchor='T.ptx["tcgen05.mma.cta_group::1.kind::f16.collector::a::use"]',
        lanes=0x1,
        warp=0,
    )


def test_raw_mma_fill_then_use_is_accepted():
    """Positive control: without the typed gemm in between, the fill stays
    valid and the ``use`` runs."""
    result = _run(_typed_gemm_between_fill_and_use(False), {"out": np.zeros((128, 16), np.float32)})
    assert result.status == {"kind": "completed"}, result.status
