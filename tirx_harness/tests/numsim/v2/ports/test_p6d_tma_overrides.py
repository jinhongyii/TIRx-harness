"""v2 port of ``tests/numsim/runtime/test_tma_overrides.py::test_tma_override_data_and_descriptor_isolation``
(all 35 ``route``/``form`` params).

Triage class ``port`` (``scripts/numsim-v2/coverage/other_assertion_triage.tsv``).
Under ``NUMSIM_IMPL=v2`` the legacy function fails only on two harness pins,
not on engine behaviour:

- ``inputs["replacement"]`` / ``inputs["input_map"]`` compared after the run:
  legacy ``Engine.run`` wrote outputs back into the caller's arrays; v2
  ``Engine.run`` never mutates its inputs (same ruling as the
  ``test_global_alias_artifact`` port). The copy asserts the redirected
  store / reduce through ``result.outputs["replacement"]`` instead, and that
  the caller's arrays are unchanged.
- ``call_op_names(module.spec.kernels[0])`` (legacy ``analyze`` source map,
  ``tests/numsim/support/manifest.py`` imports legacy internals): the copy
  checks that the parsed PrimFunc still carries the
  ``override::global_address`` qualifier on its ``T.ptx.cp`` call
  (``kernel.script()``).

Kept: clean synccheck and racecheck verdicts, the replacement/output data for
every route, and the TensorMap base output staying ``-10`` for stores. The
kernel builder and input builder are copied verbatim (``TensorMap`` from the
public package).
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine

ROUTES = ("g2cta", "g2cluster", "s2g", "reduce", "prefetch", "evict_last", "priority")
FORMS = ("address", "dim_b8", "dim_b16", "stride_b8", "stride_b16")
OVERRIDE_CASES = tuple((route, form) for route in ROUTES for form in FORMS)


def override_kernel(
    route,
    form,
    *,
    offset=4,
    coordinate=0,
    dimension=4,
    upper=0,
    issue=True,
    read_early=False,
    report=None,
):
    rank = 2 if "stride" in form or form == "address" else 1
    size = 4 if rank == 1 else 8
    load = route.startswith("g2")
    assert report is None or load
    store = route in {"s2g", "reduce"}
    tokens = ["cp", *(["reduce"] if route == "reduce" else []), "async", "bulk"]
    if route in {"prefetch", "evict_last"}:
        tokens.append("prefetch")
    if route == "priority":
        tokens = ["applypriority", "async", "bulk"]
    tokens += ["tensor", f"{rank}d"]
    if load:
        tokens += [
            "shared::cta" if route == "g2cta" else "shared::cluster",
            "global",
            "mbarrier::complete_tx::bytes",
        ]
        if report is not None:
            tokens.append(f"mbarrier::report::{report}")
    elif store:
        tokens += ["global", "shared::cta"]
        if route == "reduce":
            tokens.append("add")
        tokens.append("bulk_group")
    elif route == "priority":
        tokens += ["global", "bulk_group", "L2::evict_normal"]
    else:
        tokens += ["L2", "global"]
        if route == "evict_last":
            tokens.append("L2::evict_last")
    tokens.append("override::global_address")
    if form != "address":
        tokens.append("override::global_dim_stride" if rank == 2 else "override::global_dim")
    args = ["T.address_of(input_map)", f'T.reinterpret("uint64", replacement.ptr_to([{offset}]))']
    if form != "address":
        dtype = "uint8" if form.endswith("b8") else "uint16"
        args += [f'T.cast({dimension}, "{dtype}")']
        if rank == 2:
            args += [f'T.cast(2, "{dtype}")', "T.uint32(4)", f"T.uint16({upper})"]
    args += [f"T.int32({coordinate})", *(["T.int32(0)"] if rank == 2 else [])]
    if load:
        args = ["shared.ptr_to([0])", *args, "barrier.ptr_to([0])"]
    elif store:
        args.append("shared.ptr_to([0])")
    args.append(f"pred=T.bool({issue})")
    spelling = ".".join(tokens)
    action = f'T.ptx["{spelling}"]({", ".join(args)})'
    completion = ""
    if load:
        completion = (
            f"""
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), {size * 4})
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
"""
            if issue
            else ""
        )
    elif store or route == "priority":
        completion = """
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
"""
    if read_early:
        completion = "        T.ptx.cp.async_.bulk.commit_group()"
    if report is not None and issue:
        completion += f"""
        T.ptx.mbarrier.test_wait.parity.phase_type__primary.shared.b64(
            ready[0], reported[0], barrier.ptr_to([0]), T.uint32(0))
        output[{size}] = T.Cast("float32", ready[0])
        output[{size + 1}] = T.Cast("float32", reported[0])
"""
    observed = "replacement[4 + lane]" if read_early else "shared[lane]"
    drain = "    if lane == 0:\n        T.ptx.cp.async_.bulk.wait_group(0)" if read_early else ""
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(input_map: T.TensorMap(), replacement: T.Buffer((32772,), "float32"),
           output: T.Buffer(({size + 2 if report is not None else size},), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer(({size},), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    {'ready = T.alloc_local((1,), "uint32")' if report is not None else ""}
    {'reported = T.alloc_local((1,), "uint32")' if report is not None else ""}
    if lane < {size}:
        shared[lane] = T.cast(lane + 1, "float32")
    if lane == 0:
        T.ptx["mbarrier.init{".layout::v1" if report is not None else ""}.shared.b64"](barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        {action}
{completion}
    T.cuda.cta_sync()
    if lane < {size}:
        output[lane] = {observed}
{drain}
""",
        {"T": T},
    )


def override_inputs(form):
    rank = 2 if "stride" in form or form == "address" else 1
    original = np.full(64, -10, np.float32)
    descriptor = TensorMap(
        base=original,
        global_shape=(8,) * rank,
        global_strides=() if rank == 1 else (32,),
        box_shape=(4,) if rank == 1 else (4, 2),
        element_strides=(1,) * rank,
    ).numpy()
    return {
        "input_map": descriptor,
        "replacement": np.arange(32772, dtype=np.float32),
        "output": np.zeros(4 if rank == 1 else 8, np.float32),
    }, original


@pytest.mark.parametrize("route,form", OVERRIDE_CASES)
def test_tma_override_data_and_descriptor_isolation(route, form):
    """Replaces ``tests/numsim/runtime/test_tma_overrides.py::test_tma_override_data_and_descriptor_isolation``
    (port; see the module docstring for the dropped pins)."""
    kernel = override_kernel(route, form)
    inputs, original = override_inputs(form)
    descriptor_before = inputs["input_map"].copy()
    before = inputs["replacement"].copy()
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(kernel, {name: value.copy() for name, value in inputs.items()})
        assert report.verdict == "clean", report.format()
    result = v2.Engine().run(v2.transpile(kernel), inputs)
    indices = np.arange(4) + 4
    if inputs["output"].size == 8:
        indices = np.concatenate((indices, indices + (16 if "stride" in form else 8)))
    payload = np.arange(1, len(indices) + 1, dtype=np.float32)
    expected = before.copy()
    if route in {"s2g", "reduce"}:
        expected[indices] = payload + (before[indices] if route == "reduce" else 0)
        np.testing.assert_array_equal(result.outputs["replacement"], expected)
        # The override redirected the store away from the descriptor's own base.
        np.testing.assert_array_equal(result.outputs["input_map"], np.float32(-10))
    elif "replacement" in result.outputs:
        np.testing.assert_array_equal(result.outputs["replacement"], expected)
    np.testing.assert_array_equal(
        result.outputs["output"], before[indices] if route.startswith("g2") else payload
    )
    # v2 never mutates the caller's arrays.
    np.testing.assert_array_equal(inputs["replacement"], before)
    np.testing.assert_array_equal(inputs["input_map"], descriptor_before)
    np.testing.assert_array_equal(original, np.full(64, -10, np.float32))
    assert "override::global_address" in kernel.script()
