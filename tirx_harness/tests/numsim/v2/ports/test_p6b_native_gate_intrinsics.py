"""v2 port of the legacy Synccheck scalar gate intrinsic test.

Legacy: tests/analysis_tools/synccheck/test_native_gate_intrinsics.py::test_native_synccheck_accepts_scalar_gate_intrinsics
ran the private ``tirx_harness.numsim.checkers._run_synccheck`` on a kernel
using ``abs``/``log1p``/``exp``/``sigmoid`` gate expressions and required a
clean report, no findings, and a clean native phase with no incomplete
records. The v2 copy runs ``v2.synccheck`` on the same kernel and inputs and
asserts the same on the report and its single phase. Dropped: the
``cache_dir`` and ``max_workers`` arguments (legacy engine knobs) and the
legacy ``to_dict()["native"]`` payload shape.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import (
    assert_clean,
    assert_no_incomplete,
    requires_v2_engine,
)
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


# --- copied from tests/analysis_tools/synccheck/test_native_gate_intrinsics.py ---

@T.prim_func
def gate_intrinsics(output: T.Buffer((1,), "float32")):
    T.device_entry()
    T.cta_id([1])
    lane = T.thread_id([32])

    if lane == 0:
        value: T.float32 = T.float32(-0.5)
        output[0] = T.abs(value) + T.log1p(T.exp(value)) + T.sigmoid(value)


def test_native_synccheck_accepts_scalar_gate_intrinsics():
    """Port of tests/analysis_tools/synccheck/test_native_gate_intrinsics.py::test_native_synccheck_accepts_scalar_gate_intrinsics.

    Dropped: ``cache_dir``/``max_workers`` and the ``native`` payload key.
    """

    report = v2.synccheck(gate_intrinsics, {"output": np.zeros((1,), dtype=np.float32)})
    assert_clean(report)
    assert_no_incomplete(report)
    assert len(report.phases) == 1
    phase = report.phases[0].to_dict()
    assert phase["verdict"] == "clean"
    assert phase["incomplete"] == []
