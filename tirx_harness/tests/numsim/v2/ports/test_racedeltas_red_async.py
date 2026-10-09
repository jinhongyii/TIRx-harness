"""v2 copy of ``tests/numsim/runtime/test_red_async.py::test_red_async`` with
racecheck delta **B7**.

CTA 0 announces the bytes with a qualifier-less
``mbarrier.arrive.expect_tx.shared::cluster`` on CTA 1's barrier (``mapa``),
then issues ``red.async ... mbarrier::complete_tx::bytes`` into CTA 1's
``destination``; CTA 1 waits on the barrier (``mbarrier_wait``, ``.acquire.cta``)
and reads ``destination``. Legacy required both checkers clean. Under B7 the
arrive defaults to ``.release.cta``, which does not include CTA 1's waiter:
Racecheck reports exactly one ``scope_mismatch`` (release warp 0 -> acquire
warp 1). There is no follow-on race: the reduced bytes reach the waiter
through the complete-tx edge (delta B3: complete-tx releases at ``.cluster``
and an acquire wait of any scope receives the operation's own bytes), and
CTA 1's own initial stores are ordered by ``cluster_sync``. Synccheck stays
clean and every numeric result is unchanged. The kernel is copied verbatim
from the legacy file.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

from ._racedeltas import assert_b7_scope_mismatch

pytestmark = requires_v2_engine


def reduction_source(op, ptx_type, *, bulk=False, wait=True, lanes=1):
    dtype = {"u32": "uint32", "s32": "int32", "u64": "uint64", "b32": "uint32"}[ptx_type]
    byte_len = np.dtype(dtype).itemsize
    elements = 16 // byte_len if bulk else 1
    instruction = (
        f'T.ptx["cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.{op}.{ptx_type}"](remote_destination[0], source.ptr_to([0]), T.uint32(16), remote_barrier[0])'
        if bulk
        else f'T.ptx["red_async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.{op}.{ptx_type}"](remote_destination[0], T.cast(3, "{dtype}"), remote_barrier[0])'
    )
    return f'''
@T.prim_func
def kernel(out: T.Buffer(({elements},), "{dtype}")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.alloc_buffer(({elements},), "{dtype}", scope="shared", align=16)
    source = T.alloc_buffer(({elements},), "{dtype}", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        for i in T.unroll({elements}):
            destination[i] = T.cast(7, "{dtype}")
            source[i] = T.cast(3, "{dtype}")
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), {lanes})
    if {bulk}:
        T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane < {lanes}):
        remote_barrier = T.alloc_local((1,), "uint32")
        remote_destination = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])), T.uint32(1))
        T.ptx.mapa.shared__cluster.u32(remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])), T.uint32(1))
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(remote_barrier[0], T.uint32({elements * byte_len}), pred=True)
        {instruction}
    if (cta == 1) and (lane == 0):
        if {wait}:
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        for i in T.unroll({elements}):
            out[i] = destination[i]
    T.cuda.cluster_sync()
'''


@pytest.mark.parametrize(
    "op,ptx_type,expected",
    [
        ("add", "u32", 10),
        ("add", "s32", 10),
        ("add", "u64", 10),
        ("min", "u32", 3),
        ("min", "s32", 3),
        ("max", "u32", 7),
        ("max", "s32", 7),
        ("and", "b32", 3),
        ("or", "b32", 7),
        ("xor", "b32", 4),
        ("inc", "u32", 0),
        ("dec", "u32", 3),
    ],
)
def test_red_async(op, ptx_type, expected, tmp_path):
    """Replaces ``tests/numsim/runtime/test_red_async.py::test_red_async`` (all 12 params).

    Legacy ``run_checked``: both checkers clean, then the output. Racecheck
    delta B7: Racecheck is exactly one ``scope_mismatch`` between CTA 0's
    qualifier-less remote ``mbarrier.arrive.expect_tx.shared::cluster`` and
    CTA 1's ``mbarrier_wait``. Synccheck clean and the output are unchanged.
    """

    dtype = {"u32": "uint32", "s32": "int32", "u64": "uint64", "b32": "uint32"}[ptx_type]
    source = reduction_source(op, ptx_type)
    kernel = tvm.script.from_source(source, {"T": T})

    def args():
        return {"out": np.zeros(1, dtype=dtype)}

    v2.synccheck(kernel, args()).require_clean()
    assert_b7_scope_mismatch(v2.racecheck(kernel, args()), acquire_warps={1}, count=1, source=source)
    result = v2.Engine().run(v2.transpile(kernel, cache_dir=tmp_path), args())
    np.testing.assert_array_equal(result.outputs["out"], [expected])
