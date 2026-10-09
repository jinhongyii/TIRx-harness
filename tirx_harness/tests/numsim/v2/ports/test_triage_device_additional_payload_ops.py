"""v2 delta copy of the ``raw_tcgen_mma_tf32_ts_predicated`` item of
``tests/analysis_tools/synccheck/runtime/test_device_additional_payload_ops.py::test_payload_runtime``
(the one item of that function that fails with an "other assertion": legacy
expects ``execution_error is None`` and a clean verdict).

The other items are covered by ``test_internals_device_additional_payload_ops.py``,
which keeps this item as an ``xfail`` (``v2_gap``); this file records why v2's
rejection is the intended behaviour.

Delta: ``CONTRACT_REQUESTS.md``, "Contract changes for W1", Batch 2 item 17
("TMEM buffers (C.4 row 1)"): a ``Space::Tmem`` buffer ``Load``/``Store`` runs as
``tcgen05.ld/st .32x32b`` and "each active lane addresses its own warp
sub-partition. Anything else fails closed." The kernel runs one warp (warp 0,
TMEM lanes 0..31) but stores ``tmem[physical_lane, ...]`` with
``physical_lane = (row // 16) * 32 + row % 16`` up to 111, so lane 16 of warp 0
addresses TMEM lane 32. Legacy accepted it (clean); hardware ``tcgen05.st`` from
warp 0 cannot reach lane 32. v2 rejects it as ``bad_address`` in every mode.
"""

from __future__ import annotations

import pytest

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tests.numsim.v2.ports.test_internals_device_additional_payload_ops import (
    _arguments,
    raw_tcgen_mma_tf32_ts_predicated,
)
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_MESSAGE = "tmem lane 32 is outside warp 0's sub-partition"


def test_payload_runtime_raw_tcgen_mma_tf32_ts_predicated_rejects_cross_subpartition_tmem_store():
    """Delta copy (CONTRACT_REQUESTS Batch 2 item 17); see the module docstring."""
    arguments = _arguments("raw_tcgen_mma_tf32_ts_predicated")
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(raw_tcgen_mma_tf32_ts_predicated, arguments)
        assert report.verdict == "error", report.format()
        assert [(f.status, f.kind) for f in report.findings] == [("error", "bad_address")]
        assert _MESSAGE in report.findings[0].message
        assert report.findings[0].details["lanes"] == "WarpMask(0x00010000)"  # lane 16 -> row 16
    with pytest.raises(v2.ExecutionError, match="outside warp 0's sub-partition"):
        v2.Engine().run(v2.transpile(raw_tcgen_mma_tf32_ts_predicated), arguments)
