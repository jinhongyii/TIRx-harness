"""v2 copy of ``tests/numsim/runtime/test_pointer_slot_arrays.py::
test_pointer_array_initialization_and_bounds`` asserting kinds, not text.

- ``valid``: unchanged (both checkers clean, outputs equal the source). The
  predicated-off ``ld`` of ``pointers[99]`` (``pred=False``) evaluates no
  operand, so it is not an out-of-bounds access.
- ``oob``: ``pointers[2]`` indexes past the two-slot local array. Both checkers
  report verdict ``error`` with the single error kind ``out_of_bounds`` (legacy
  ``oob``, racecheck delta T6) at that load, and ``Engine.run`` raises
  ``NumSimExecutionError`` whose stopping diagnostic is ``out_of_bounds``;
  legacy pinned the text ``outside|exceeds`` (test-migration.md, "Public-API A
  tests under ``NUMSIM_IMPL=v2``": same exception, only the wording differs).
- ``uninitialized``: each lane stores only ``pointers[lane % 2]`` and loads
  ``pointers[1 - lane % 2]``, a slot of its own local array it never wrote: a
  genuinely uninitialized register read. W8-5 (``ValidityPolicy::ZeroAndReport``
  in every mode) and V2C-20 (register-space tracking) report it as
  ``uninitialized_read``/``review`` and read it as zero; the resulting null
  global address is the error. Legacy named that error ``oob`` with the text
  ``out-of-bounds``; v2 names it ``bad_address`` (unmapped global address 0x0,
  a runtime kind per racecheck delta T6), so the copy asserts that kind.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import NumSimExecutionError
from tests.numsim.v2.checkers._runnable import requires_v2_engine

pytestmark = requires_v2_engine


def partial_array_case(mode):
    index = {"valid": "lane % 2", "uninitialized": "1 - lane % 2", "oob": "2"}[mode]
    source = f"""
@T.prim_func
def kernel(source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointers = T.alloc_local((2,), "uint64")
    pointers[lane % 2] = T.reinterpret("uint64", source.ptr_to([lane]))
    T.cuda.cta_sync()
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", pointers[{index}]))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", pointers[99]), pred=False, preserve_dst=True)
"""
    return tvm.script.from_source(source, {"T": T}), source


def _span_text(source: str, record: dict) -> str:
    span = record.get("source_span") or {}
    line = source.split("\n")[span["line"] - 1]
    return line[span["column"] - 1 : span["end_column"] - 1]


def _first_stop(error) -> dict:
    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


@pytest.mark.parametrize("mode", ["valid", "uninitialized", "oob"])
def test_pointer_array_initialization_and_bounds(mode, tmp_path):
    kernel, source = partial_array_case(mode)
    inputs = {"source": np.arange(32, dtype=np.uint32) + 100, "output": np.zeros(32, np.uint32)}
    error_kind = {"uninitialized": "bad_address", "oob": "out_of_bounds"}.get(mode)
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(kernel, {k: v.copy() for k, v in inputs.items()})
        if mode == "valid":
            report.require_clean()
            continue
        assert report.verdict == "error", report.format()
        errors = [f for f in report.findings if f.status == "error"]
        assert [f.kind for f in errors] == [error_kind], report.format()
        if mode == "oob":
            assert _span_text(source, errors[0].details) == "pointers[2]", errors[0].details
            assert {f.status for f in report.findings} == {"error"}, report.format()
        else:
            # The integer model materializes the uninitialized word as zero;
            # the review (W8-5 / V2C-20) and the later invalid-address error remain.
            reviews = [f for f in report.findings if f.status != "error"]
            assert reviews, report.format()
            for finding in reviews:
                assert (finding.status, finding.kind) == ("review", "uninitialized_read"), report.format()
                assert finding.details.get("space") == "reg", finding.details
                assert _span_text(source, finding.details) == "pointers[1 - lane % 2]", finding.details
    module = v2.transpile(kernel, cache_dir=tmp_path)
    if mode == "valid":
        result = v2.Engine().run(module, inputs)
        np.testing.assert_array_equal(result.outputs["output"], inputs["source"])
    else:
        with pytest.raises(NumSimExecutionError) as excinfo:
            v2.Engine().run(module, inputs)
        assert isinstance(excinfo.value, v2.ExecutionError), type(excinfo.value)
        stop = _first_stop(excinfo.value)
        assert (stop["status"], stop["kind"]) == ("error", error_kind), stop
        if mode == "uninitialized":
            assert any(
                (d.get("status"), d.get("kind")) == ("review", "uninitialized_read")
                for d in excinfo.value.diagnostics
            ), excinfo.value.diagnostics
