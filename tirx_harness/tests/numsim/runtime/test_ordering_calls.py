from __future__ import annotations

import numpy as np
import pytest
import tvm

from tirx_harness import numsim
from tests.numsim.support.kernels import ordering_only_control_calls
from tvm.script import tirx as T


@T.prim_func
def setmaxnreg_ordering_only(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    T.cuda.warpgroup_sync(7)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(256)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_without_intervening_sync(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(256)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_warp_disagreement(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    if warp == 0:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    else:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(32)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_equivalent_branch_sites(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    if warp == 0:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    elif warp == 1:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    elif warp == 2:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    else:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    if (warp == 0) and (lane == 0):
        output[0] = 11


@T.prim_func
def setmaxnreg_deleted(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.cuda.warpgroup_sync(7)
    if (warp == 0) and (lane == 0):
        output[0] = 7


def _setmaxnreg_static_expression_kernel():
    return tvm.script.from_source(
        """
@T.prim_func
def setmaxnreg_static_expressions(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(T.int32(24) + T.int32(8))
    T.cuda.warpgroup_sync(7)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(T.int32(72) - T.int32(8))
    if (warp == 0) and (lane == 0):
        output[0] = 1
""",
        extra_vars={"T": T},
    )


@T.prim_func
def divergent_warp_sync(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.cuda.warp_sync()
    output[lane] = lane


@T.prim_func
def griddep_producer(intermediate: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    intermediate[lane] = lane + 1
    T.ptx.griddepcontrol.launch_dependents()


@T.prim_func
def griddep_consumer(intermediate: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.griddepcontrol.wait()
    output[lane] = intermediate[lane] * 2


@pytest.mark.parametrize(
    ("statement", "message"),
    (
        (
            "T.ptx.fence.mbarrier_init.release.cluster(T.int32(0))",
            "fence_mbarrier_init expects 0 operand",
        ),
        (
            "T.ptx.griddepcontrol.wait(T.int32(0))",
            "griddepcontrol expects 0 operand",
        ),
    ),
)
def test_target_ptx_parser_rejects_extra_ordering_operands(statement, message):
    source = f"""
@T.prim_func
def invalid():
    T.device_entry()
    {statement}
"""
    with pytest.raises(tvm.error.DiagnosticError, match=message):
        tvm.script.from_source(source, {"T": T})


@pytest.mark.parametrize(
    ("parameters", "register_count", "message"),
    (
        (
            "nreg: T.int32",
            "nreg",
            "compile-time integer constant, got Var",
        ),
        (
            "",
            'T.Cast("int32", T.int64(64))',
            "compile-time integer constant, got Cast",
        ),
        (
            "",
            "16",
            r"must be one of .* got 16",
        ),
        (
            "",
            "25",
            r"must be one of .* got 25",
        ),
        (
            "",
            "264",
            r"must be one of .* got 264",
        ),
    ),
)
def test_setmaxnreg_parser_rejects_nonliteral_and_illegal_counts(
    parameters, register_count, message
):
    source = f"""
@T.prim_func
def invalid({parameters}):
    T.device_entry()
    T.ptx.setmaxnreg.inc.sync.aligned.u32({register_count})
"""
    with pytest.raises(tvm.error.DiagnosticError, match=message):
        tvm.script.from_source(source, {"T": T})


def test_setmaxnreg_accepts_equivalent_requests_from_distinct_branch_sites(tmp_path):
    module = numsim.transpile(setmaxnreg_equivalent_branch_sites, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([11], dtype=np.int32))


