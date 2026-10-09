"""v2 copies of the padded-TMA tests in ``tests/numsim/integration/test_reported_layout_regressions.py`` (row L7).

Row L7 (``docs/development/numsim-behaviour-deltas.md``, "TMA into a padded
shared layout" in ``numsim-isa-answers.md``): a tile ``copy_async`` TMA into a
padded shared slice (``tma_padded_narrow_rows``) cannot be one TMA box. Legacy
compiled it and rejected it at run time ("TMA shared payload component 1 ...
128-byte aligned"); v2 rejects it at transpile time with
``UnsupportedTIRxError``. Only the phase changes: the kernel still fails
closed. The checkers run the same transpile, so they return a fail-closed
``incomplete`` report (reason ``native_frontend_unsupported``), never a clean
one. The kernel is copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.tirx.lang.pipeline import TMABar

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine

LANES = 32


@T.prim_func
def tma_padded_narrow_rows(source: T.Buffer((64, 8), "bfloat16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    pool = T.SMEMPool()
    ready = TMABar(pool, 1)
    ready.init(1)
    pool.move_base_to(1024)
    shared = pool.alloc(
        (64, 16),
        "bfloat16",
        layout=mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, (64, 16)),
        align=1024,
    )
    pool.commit()

    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if T.cuda.elect_sync():
        Tx.copy_async(
            shared[:, 0:8],
            source[:, :],
            dispatch="tma_auto",
            mbar=ready.ptr_to([0]),
        )
        ready.arrive(0, tx_count=64 * 8 * 2)
    ready.wait(0, 0)


def _inputs():
    return {"source": np.zeros((64, 8), dtype=np.uint16)}


def test_numsim_rejects_a_misaligned_tma_shared_component():
    """Replaces ``tests/numsim/integration/test_reported_layout_regressions.py::test_numsim_rejects_a_misaligned_tma_shared_component``.

    Row L7 (phase change): legacy raised ``NumSimExecutionError`` ("component
    1 byte offset 1056 must be 128-byte aligned") at run time; v2 raises
    ``UnsupportedTIRxError`` at ``v2.transpile``. The kernel shape is unchanged."""

    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(tma_padded_narrow_rows)


@pytest.mark.parametrize(
    "checker",
    [pytest.param("synccheck", id="synccheck"), pytest.param("racecheck", id="racecheck")],
)
def test_checkers_reject_a_misaligned_tma_shared_component(checker):
    """Replaces ``tests/numsim/integration/test_reported_layout_regressions.py::test_checkers_reject_a_misaligned_tma_shared_component`` (both params).

    Row L7 (phase change): legacy reported verdict ``error`` with the single
    finding kind ``tma_shared_address_misaligned``. v2 rejects the kernel at
    transpile (``UnsupportedTIRxError``), so the checker fails closed with an
    ``incomplete`` report (reason ``native_frontend_unsupported``); it is
    never clean."""

    with pytest.raises(UnsupportedTIRxError):
        v2.transpile(tma_padded_narrow_rows)
    report = getattr(v2, checker)(tma_padded_narrow_rows, _inputs())
    assert report.verdict == "incomplete", report.format()
    (phase,) = report.phases
    reasons = [record.get("reason") for record in phase.to_dict()["incomplete"]]
    assert reasons == ["native_frontend_unsupported"], phase.to_dict()
