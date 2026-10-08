"""v2 port of ``tests/numsim/integration/test_written_buffer_inference.py::test_dynamic_raw_write_requires_bound_address``."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def unresolved_raw_store(pointer_bits: T.Buffer((1,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.st.global_.u32(T.reinterpret("handle", pointer_bits[0]), T.uint32(1))


def _first_stop(error: v2.ExecutionError) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


@pytest.mark.parametrize(
    "address_kind",
    [
        pytest.param(
            "bound",
            marks=v2_gap(
                "a raw host address (ctypes.data of the bound pointer_bits input) is not mapped: "
                "bad_address error in NumSim/racecheck/synccheck; expected the store to land in "
                "pointer_bits (test-migration.md: raw host addresses unportable until v2 exposes "
                "a binding's engine address)"
            ),
        ),
        "null",
        pytest.param(
            "unbound",
            marks=v2_gap(
                "an integer address naming no binding (0x1000) is an error bad_address; "
                "expected verdict incomplete (integer_address_without_binding)"
            ),
        ),
    ],
)
def test_dynamic_raw_write_requires_bound_address(address_kind):
    """Port of ``tests/numsim/integration/test_written_buffer_inference.py::test_dynamic_raw_write_requires_bound_address``.

    ``analyze``/``verify`` -> a successful ``v2.transpile``. Checker verdicts
    through ``v2.racecheck`` / ``v2.synccheck``. Dropped: the legacy message
    text ("null pointer", "integer_address_without_binding"); the null case
    asserts the error verdict and an error stop, not the kind name (v2 reports
    ``bad_address``; no delta row names the legacy kind)."""

    module = v2.transpile(unresolved_raw_store)
    bits = np.zeros(1, np.uint64)
    bits[0] = {"bound": bits.ctypes.data, "null": 0, "unbound": 0x1000}[address_kind]
    inputs = {"pointer_bits": bits}
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(unresolved_raw_store, inputs)
        if address_kind == "bound":
            assert report.verdict == "clean", report.format()
        else:
            assert report.verdict == ("error" if address_kind == "null" else "incomplete"), report.format()
    if address_kind == "bound":
        result = v2.Engine().run(module, inputs)
        assert set(result.outputs) == {"pointer_bits"}
        expected = (int(bits[0]) & 0xFFFFFFFF00000000) | 1
        np.testing.assert_array_equal(result.outputs["pointer_bits"], [expected])
    else:
        with pytest.raises(v2.ExecutionError) as excinfo:
            v2.Engine().run(module, inputs)
        stop = _first_stop(excinfo.value)
        assert stop["status"] == ("error" if address_kind == "null" else "incomplete"), stop
