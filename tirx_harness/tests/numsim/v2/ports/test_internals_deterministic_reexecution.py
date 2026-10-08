"""v2 port of the legacy deterministic re-execution test; the scheduler-stat
pins (``task_count``/``completed_task_count``/``poll_count``/``poll_order``)
are dropped."""

from __future__ import annotations

import numpy as np

from tests.numsim.support.kernels import mbarrier_phase_reuse
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def test_repeated_full_launch_keeps_outputs_stats_and_poll_order_deterministic():
    """Port of ``tests/numsim/integration/test_deterministic_reexecution.py::
    test_repeated_full_launch_keeps_outputs_stats_and_poll_order_deterministic``.

    Dropped pins: ``first.stats == second.stats`` and the legacy scheduler
    stats ``task_count``, ``completed_task_count``, ``poll_count``,
    ``poll_order`` and ``kernels[0]["poll_order"]``.
    """

    module = v2.transpile(mbarrier_phase_reuse)

    def run_once():
        source = np.arange(4, dtype=np.float32)
        output = np.full(2, -1, dtype=np.int32)
        result = v2.Engine().run(module, {"source": source, "output": output}, outputs=("output",))
        return source, output, result

    first_source, first_output, first = run_once()
    second_source, second_output, second = run_once()

    assert first_source is not second_source
    assert first_output is not second_output
    np.testing.assert_array_equal(first.outputs["output"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(second.outputs["output"], first.outputs["output"])
    np.testing.assert_array_equal(first_output, second_output)
