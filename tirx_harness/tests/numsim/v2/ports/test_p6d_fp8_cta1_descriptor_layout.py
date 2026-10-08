"""v2 expected-error port of
``tests/numsim/integration/test_fp8_cta1_descriptor_layout.py::test_fp8_cta1_extended_shared_addresses``
(both params, ``ss`` and ``ts``).

Ruling: ``docs/development/numsim-behaviour-deltas.md`` row T4 (and
``CONTRACT_REQUESTS.md`` "Contract changes for W1", Batch 2 item 17; W2
engine-stops triage, "TMEM buffer access outside the warp's sub-partition").
The kernel runs one warp (warp 0, TMEM lanes 0..31) but its buffer-form TMEM
accesses reach lane 32: the ``ts`` variant fills ``tmem[rg * 32 + lane, ...]``
for ``rg`` up to 3, and both variants read back
``tmem[(row // 16) * 32 + row % 16, col]`` for rows up to 63. A buffer-form
TMEM ``Load``/``Store`` runs as ``tcgen05.ld/st .32x32b``, which can only
address the warp's own sub-partition, so NumSim, Synccheck and Racecheck all
stop with ``bad_address`` (``tmem[i]: tmem lane 32 is outside warp 0's
sub-partition``). Legacy modelled TMEM buffers abstractly, returned 32.0 in
every output element and reported both checkers clean.

The legacy tail (``sm_100a`` with the shared offset ``1 << 18``) raised
``UnsupportedTIRxError`` "exceeds the 18-bit descriptor address space" at
transpile. v2 also rejects it at transpile, for a different reason: its
272384-byte shared window is above the 227 KB per-CTA capacity of ``sm_100a``
(delta F5; such a CTA cannot launch). The descriptor start-address field is
still decoded per arch at run time (``interp/handlers/tcgen.rs``, ``TcArch``).

The kernel builder is copied verbatim from the legacy module.
"""

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TLane, TileLayout

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine

_TMEM = TileLayout(S[(128, 32) : (1 @ TLane, 1 @ TCol)])


def _kernel(offset, a_in_tmem):
    @T.prim_func
    def kernel(output: T.Buffer((64, 8), "uint32")):
        T.device_entry()
        lane = T.thread_id([32])
        barrier = T.alloc_buffer((1,), "uint64", scope="shared")
        arena = T.alloc_buffer((offset + 9216,), "uint8", scope="shared", align=1024)
        a = T.decl_buffer((8192,), "uint8", data=arena.data, elem_offset=offset, scope="shared")
        b = T.decl_buffer(
            (1024,), "uint8", data=arena.data, elem_offset=offset + 8192, scope="shared"
        )
        tmem = T.decl_buffer((128, 32), "uint32", scope="tmem", layout=_TMEM, allocated_addr=0)
        da: T.uint64
        db: T.uint64
        di: T.uint32
        for i in T.serial(256):
            a[lane + i * 32] = T.uint8(0x38)  # E4M3 1.0
        for i in T.serial(32):
            b[lane + i * 32] = T.uint8(0x38)
        if a_in_tmem:
            for rg in T.unroll(4):
                for col in T.unroll(8):
                    tmem[rg * 32 + lane, 16 + col] = T.uint32(0x38383838)
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.ptx.tcgen05.fence__after_thread_sync()
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tcgen05.encode_instr_descriptor(
                T.address_of(di),
                d_dtype="float32",
                a_dtype="float8_e4m3fn",
                b_dtype="float8_e4m3fn",
                M=64,
                N=8,
                K=32,
                trans_a=False,
                trans_b=False,
                n_cta_groups=1,
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(da),
                T.address_of(arena[0]),
                ldo=0,
                sdo=64,
                swizzle=3,
            )
            # Supply the SM107 15-bit start field explicitly; do not let a
            # legacy encoder mask off the address bit under test.
            db = (da & T.bitwise_not(T.uint64(0x7FFF))) | T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(b.ptr_to([0])), T.uint32(4)), "uint64"
            )
            da = (da & T.bitwise_not(T.uint64(0x7FFF))) | T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(a.ptr_to([0])), T.uint32(4)), "uint64"
            )
            if a_in_tmem:
                T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                    T.uint32(0),
                    T.uint32(16),
                    db,
                    di,
                    0,
                    0,
                    0,
                    0,
                    T.ptx.pred(0),
                )
            else:
                T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                    T.uint32(0),
                    da,
                    db,
                    di,
                    0,
                    0,
                    0,
                    0,
                    T.ptx.pred(0),
                )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barrier[0])
            )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for rg in T.unroll(2):
            row = rg * 32 + lane
            for col in T.unroll(8):
                output[row, col] = tmem[(row // 16) * 32 + row % 16, col]

    return kernel


def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


@pytest.mark.parametrize("a_in_tmem", [False, True], ids=["ss", "ts"])
def test_fp8_cta1_extended_shared_addresses(a_in_tmem):
    """Replaces the legacy function of the same name (both params); delta T4, see the module docstring."""

    buffer_index = "tmem[1040]" if a_in_tmem else "tmem[1024]"
    for offset in (0, 1 << 18):
        kernel = _kernel(offset, a_in_tmem).with_attr("tirx.cuda_arch", "sm_107a")
        arguments = {"output": np.zeros((64, 8), dtype=np.uint32)}
        with pytest.raises(v2.ExecutionError) as excinfo:
            v2.Engine(max_workers=1).run(v2.transpile(kernel), arguments)
        stop = _first_stop(excinfo.value)
        assert stop["status"] == "error", stop
        assert stop["kind"] == "bad_address", stop
        assert f"{buffer_index}: tmem lane 32 is outside warp 0's sub-partition" in str(
            excinfo.value
        )
        for checker in (v2.synccheck, v2.racecheck):
            report = checker(kernel, arguments)
            assert report.verdict == "error", report.format()
            assert [(f.status, f.kind) for f in report.findings] == [("error", "bad_address")]
            assert "outside warp 0's sub-partition" in report.findings[0].message

    # On sm_100a the 272384-byte shared window is above the per-CTA capacity:
    # rejected at transpile (legacy also rejected it at transpile).
    with pytest.raises(UnsupportedTIRxError, match="above the 232448-byte per-CTA capacity of sm_100a"):
        v2.transpile(kernel.with_attr("tirx.cuda_arch", "sm_100a"))
