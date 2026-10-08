"""v2 port of the legacy TMA-override invalid-operand test.

The legacy ``analyze(kernel).kernels[0].source_map`` walk (to find the
``op_id`` of the override call) becomes a check on the finding's v2 source
operation text; the legacy message substrings are not pinned. The kernel
builder and inputs are copied verbatim from
``tests/numsim/runtime/test_tma_overrides.py`` (``numsim.TensorMap`` is the
surviving ``tirx_harness.numsim.TensorMap``).
"""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


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




@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"offset": 1}, id="unaligned_address"),
        pytest.param({"offset": 8}, id="address_window"),
        pytest.param(
            {"coordinate": 1}, id="nonzero_coordinate"
        ),
        pytest.param({"dimension": 256}, id="dimension_8bit"),
        pytest.param({"upper": 16}, id="upper_stride_bits"),
    ],
)
def test_tma_override_rejects_invalid_operands(kwargs):
    """Port of ``tests/numsim/runtime/test_tma_overrides.py::test_tma_override_rejects_invalid_operands``
    (legacy params ``16-byte aligned``, ``128 KiB``, ``zero coordinates``,
    ``8-bit``, ``unused bits``, in that order).

    Both checkers report an ``error`` verdict whose first finding is the
    ``invalid_operand`` kind on the override call (legacy compared the
    finding's ``source_op_id`` with the op id found by walking
    ``analyze(kernel).kernels[0].source_map``; v2 names the call in the
    finding's source operation). ``Engine.run`` raises ``v2.ExecutionError``
    whose stop is an ``invalid_operand`` error. The legacy message substrings
    are not pinned.
    """

    kernel = override_kernel("g2cta", "stride_b16", **kwargs)
    inputs, _ = override_inputs("stride_b16")
    module = v2.transpile(kernel)
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(kernel, inputs)
        assert report.verdict == "error", report.format()
        finding = report.findings[0]
        assert finding.kind == "invalid_operand", report.format()
        source_text = finding.details["operation"]["source"]["source_text"]
        assert "override_global_dim_stride_b16" in source_text, finding.details["operation"]
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, inputs)
    stops = [d for d in caught.value.diagnostics if d.get("status") in ("error", "incomplete")]
    assert stops and stops[0]["status"] == "error" and stops[0]["kind"] == "invalid_operand", stops
