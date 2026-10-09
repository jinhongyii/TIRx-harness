"""v2 port of ``tests/numsim/integration/test_emission_errors.py::test_native_topology_returns_semantic_errors_as_data``.

Legacy called the private ``native_frontend._library()["numsim_launch_topology"]``
service and ``native_frontend.launch_topology``. The observable contract kept
here: a ``thread_id_in_wg`` extent other than 128 is rejected, and an extent
of 128 yields a four-warp CTA (the v2 document topology). Dropped: the
``{"value", "error"}`` data shape of the native service and the message.
"""

from __future__ import annotations

import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import NumSimBuildError, UnsupportedTIRxError

pytestmark = requires_v2_engine


def _kernel(threads: int):
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    thread = T.thread_id_in_wg([{threads}])
    T.evaluate(thread)
""",
        {"T": T},
    )


@pytest.mark.parametrize(
    "threads",
    [
        64,
        128,
    ],
)
def test_native_topology_returns_semantic_errors_as_data(threads):
    """Port of ``tests/numsim/integration/test_emission_errors.py::test_native_topology_returns_semantic_errors_as_data``.

    The rejection may come from ``transpile`` or from launching the module
    (as in the p6b frontend topology ports), but it must come."""

    kernel = _kernel(threads)
    if threads == 64:
        with pytest.raises((UnsupportedTIRxError, NumSimBuildError)):
            v2.Engine().run(v2.transpile(kernel), {})
    else:
        module = v2.transpile(kernel)
        block = module.document["kernels"][0]["topology"]["block"]
        assert block[0] * block[1] * block[2] // 32 == 4
        result = v2.Engine().run(module, {})
        assert result.status["kind"] == "completed", result.status
