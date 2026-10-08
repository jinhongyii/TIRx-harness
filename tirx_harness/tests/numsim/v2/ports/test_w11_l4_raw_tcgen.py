"""v2 copy of ``tests/numsim/runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_f8f6f4_cta2_sm100_rejects_sm107_k64_bit``
(W11, blocked-v2 batch).

The legacy kernel issues a raw ``tcgen05.mma.cta_group::2.kind::f8f6f4`` whose
runtime instruction descriptor ``0x10040490`` encodes M = 256, N = 16. Delta
numsim-behaviour-deltas L4 (``tcgen05.mma`` shapes): cta_group::2 needs
N in 32..=256 by 32, so v2 stops that run with ``invalid_operand`` (legacy
accepted it). The legacy test's point is the SM100-family rule for the SM107-only
K64 bit (bit 29): with it set, the descriptor must be rejected.

- ``test_..._original_shape_is_invalid``: the verbatim descriptor stops with
  ``invalid_operand`` at the MMA (delta L4).
- ``test_raw_tcgen_f8f6f4_cta2_sm100_rejects_sm107_k64_bit``: the same kernel
  with N = 32 (descriptor N field ``N >> 3`` = 4, i.e. ``0x10080490``) runs to
  completion, and the same descriptor with the K64 bit (``0x30080490``) is
  rejected with ``invalid_operand``, for every SM100-family architecture.
"""

from __future__ import annotations

import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])
_ARCHS = pytest.mark.parametrize("arch", ("sm_100a", "sm_100f", "sm_103a", "sm_103f"))


@T.prim_func
def runtime_descriptor(descriptor: T.uint32):
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
    for i in T.serial(512):
        shared_a[lane + i * 32] = T.uint8(0)
    for i in T.serial(256):
        shared_b[lane + i * 32] = T.uint8(0)
    T.cuda.cluster_sync()
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
            descriptor,
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
    T.cuda.cluster_sync()


def _stop(caught):
    stops = [d for d in caught.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops, caught.value.diagnostics
    return stops[0]


@_ARCHS
def test_raw_tcgen_f8f6f4_cta2_sm100_rejects_sm107_k64_bit_original_shape_is_invalid(arch):
    """Expected error (delta L4): M = 256, N = 16 is not a cta_group::2 shape."""
    module = v2.transpile(runtime_descriptor.with_attr("tirx.cuda_arch", arch))
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"descriptor": 0x10040490})
    stop = _stop(caught)
    assert (stop["status"], stop["kind"]) == ("error", "invalid_operand"), stop
    assert "N in 32..=256" in stop["message"], stop


@_ARCHS
def test_raw_tcgen_f8f6f4_cta2_sm100_rejects_sm107_k64_bit(arch):
    """Corrected copy: N = 32 runs; the SM107-only K64 bit is rejected on SM100/SM103."""
    module = v2.transpile(runtime_descriptor.with_attr("tirx.cuda_arch", arch))
    result = v2.Engine().run(module, {"descriptor": 0x10080490})
    assert result.status.get("kind") == "completed", result.status
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"descriptor": 0x30080490})
    stop = _stop(caught)
    assert (stop["status"], stop["kind"]) == ("error", "invalid_operand"), stop
    assert "dense F32/E5M2/E5M2" in stop["message"], stop
