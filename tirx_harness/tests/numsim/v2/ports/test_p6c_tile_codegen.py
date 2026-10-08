"""v2 ports of legacy ``tests/numsim/runtime/test_tile_codegen.py`` tests that
pinned run stats or legacy error text. Kernels come from the shared
``tests.numsim.support.kernels`` fixtures, as in the legacy file."""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.support.kernels import tma_copy_cluster_multicast, tma_copy_transaction_mismatch
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def _stops(error: v2.ExecutionError) -> list[dict]:
    return [d for d in error.diagnostics if d.get("status") in ("error", "incomplete")]


def _run_under_delivery():
    source = np.arange(32, dtype=np.float16).reshape(4, 8)
    output = np.zeros_like(source)
    module = v2.transpile(tma_copy_transaction_mismatch)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"source": source, "output": output})
    return caught.value


def test_tma_multicast_writes_each_target_cta_and_credits_each_copy():
    """Port of ``tests/numsim/runtime/test_tile_codegen.py::test_tma_multicast_writes_each_target_cta_and_credits_each_copy``.

    Dropped pin: ``result.stats["task_count"] == 2``; the run status is
    asserted ``completed`` (both target CTAs' barriers are credited, or the
    waits would never finish).
    """

    source = np.linspace(-2, 3, 8, dtype=np.float32)
    output = np.zeros((2, 8), dtype=np.float32)

    module = v2.transpile(tma_copy_cluster_multicast)
    result = v2.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.stack([source, source]))
    assert result.status.get("kind") == "completed", result.status


def test_tma_transaction_under_delivery_reports_deterministic_deadlock():
    """Port of ``tests/numsim/runtime/test_tile_codegen.py::test_tma_transaction_under_delivery_reports_deterministic_deadlock``.

    Fail-closed half: the run raises ``v2.ExecutionError`` (a
    ``NumSimExecutionError``) with a stopping diagnostic. Dropped pin: the
    legacy ``transactions=64/68`` text. Whether the stop is an ``error`` is
    :func:`test_tma_transaction_under_delivery_is_an_error`.
    """

    error = _run_under_delivery()
    assert _stops(error), error.diagnostics


@v2_gap(
    "TMA transaction under-delivery (64 of 68 expected bytes) stops as incomplete "
    "'analysis_incomplete: divergent_block: no progress while warp 1 is blocked with a "
    "divergent mask' (legacy: NumSimExecutionError deadlock error); error->incomplete, "
    "no delta row yet"
)
def test_tma_transaction_under_delivery_is_an_error():
    """Second half of ``tests/numsim/runtime/test_tile_codegen.py::test_tma_transaction_under_delivery_reports_deterministic_deadlock``:
    the stop is an ``error`` diagnostic, as legacy raised a deadlock error."""

    error = _run_under_delivery()
    assert _stops(error)[0]["status"] == "error", error.diagnostics
