"""v2 copies of the four category-A functions in
``tests/analysis_tools/shared/test_native_{dynamic_launch,predicated_reduction,shared_descriptor_table}.py``
(W11 legacy-import batch). The legacy tests called the internal
``checkers._run_synccheck`` / ``checkers._run_racecheck`` and read the legacy
``native`` payload (``phase.topology.warp_count``, ``input.bindings/scalars``,
``engine.artifact_key``, ``input.digest``, ``access_count``). The copies use the
public ``v2.racecheck`` / ``v2.synccheck`` and replace payload internals with the
observable result (NumSim outputs, verdicts, finding kind and source anchor).

- ``test_checker_binds_runtime_scalar_launch_extent``: both checkers are clean,
  and the runtime ``num_ctas`` launch extent is bound: 4 CTAs each write ``value``.
- ``test_checker_artifact_identity_tracks_launch_scalars_only``: the artifact key
  and input digest are cache internals; the observable fact is that changing the
  data scalar or the launch scalar is reflected in the result (no stale reuse).
- ``test_checker_models_predicated_global_reduction``: clean in both checkers;
  only the 16 predicated-on lanes add (legacy pinned ``access_count == 1``).
- ``test_checker_follows_matrix_descriptor_through_raw_shared_table``: delta
  numsim-behaviour-deltas T4. The MMA reads its A descriptor back from a shared
  table and completes; the first error is at the later TMEM read-back, which
  warp 0 does for TMEM lanes outside its 32-lane sub-partition (``bad_address``,
  legacy accepted it), anchored at that read.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])
_CHECKERS = pytest.mark.parametrize("check", [v2.synccheck, v2.racecheck], ids=["synccheck", "racecheck"])


@T.prim_func
def runtime_grid(num_ctas: T.int32, value: T.int32, output: T.Buffer((4,), "int32")):
    T.device_entry()
    cta = T.cta_id([num_ctas])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[cta] = value


@T.prim_func
def predicated_global_reduction(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["red.global.add.f32"](
        output.ptr_to([lane]),
        T.float32(1.0),
        pred=lane < 16,
    )


@T.prim_func
def raw_tcgen_mma_bf16_ss(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
    output_ws: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    prefix = T.alloc_buffer((128,), "uint8", scope="shared")
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    descriptor_table = T.alloc_buffer((1,), "uint64", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_a_loaded: T.uint64
    desc_b: T.uint64
    prefix[lane] = T.cast(lane, "uint8")
    for copy_i in T.serial(128):
        shared_a[lane + copy_i * 32] = a_physical[lane + copy_i * 32]
    for copy_i in T.serial(32):
        shared_b[lane + copy_i * 32] = b_physical[lane + copy_i * 32]
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=8,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx.st.shared.u64(descriptor_table.ptr_to([0]), desc_a)
        T.ptx.ld.shared.u64(desc_a_loaded, descriptor_table.ptr_to([0]))
        T.ptx["tcgen05.mma.cta_group::1.kind::f16.collector::a::discard"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", _tmem[physical_lane, col])
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.unroll(8):
            physical_lane = row + (col // 4) * 64
            physical_col = col % 4
            output_ws[row, col] = T.reinterpret("float32", _tmem[physical_lane, physical_col])


def _grid_inputs(num_ctas: int, value: int) -> dict:
    return {"num_ctas": np.int32(num_ctas), "value": np.int32(value), "output": np.zeros(4, dtype=np.int32)}


@_CHECKERS
def test_checker_binds_runtime_scalar_launch_extent(check):
    """Copy; see the module docstring."""
    check(runtime_grid, _grid_inputs(4, 7)).require_clean()
    result = v2.Engine().run(v2.transpile(runtime_grid), _grid_inputs(4, 7))
    np.testing.assert_array_equal(result.outputs["output"], [7, 7, 7, 7])


@_CHECKERS
def test_checker_artifact_identity_tracks_launch_scalars_only(check):
    """Copy; cache identity replaced by results that follow both scalars."""
    module = v2.transpile(runtime_grid)
    expected = {(2, 3): [3, 3, 0, 0], (2, 9): [9, 9, 0, 0], (4, 9): [9, 9, 9, 9]}
    for (num_ctas, value), output in expected.items():
        check(runtime_grid, _grid_inputs(num_ctas, value)).require_clean()
        result = v2.Engine().run(module, _grid_inputs(num_ctas, value))
        np.testing.assert_array_equal(result.outputs["output"], output)


@_CHECKERS
def test_checker_models_predicated_global_reduction(check):
    """Copy; see the module docstring."""
    check(predicated_global_reduction, {"output": np.zeros(32, dtype=np.float32)}).require_clean()
    result = v2.Engine().run(
        v2.transpile(predicated_global_reduction), {"output": np.zeros(32, dtype=np.float32)}
    )
    np.testing.assert_array_equal(result.outputs["output"], [1.0] * 16 + [0.0] * 16)


@_CHECKERS
def test_checker_follows_matrix_descriptor_through_raw_shared_table(check):
    """Delta copy (T4); see the module docstring."""
    report = check(
        raw_tcgen_mma_bf16_ss,
        {
            "a_physical": np.zeros(4096, dtype=np.uint8),
            "b_physical": np.zeros(1024, dtype=np.uint8),
            "output": np.zeros((64, 8), dtype=np.float32),
            "output_ws": np.zeros((64, 8), dtype=np.float32),
        },
    )
    assert report.verdict == "error", report.format()
    errors = [f for f in report.findings if f.status == "error"]
    assert [f.kind for f in errors] == ["bad_address"], report.format()
    span = errors[0].details["source_span"]
    with open(span["source_name"]) as handle:
        line = handle.read().splitlines()[span["line"] - 1]
    assert "_tmem[physical_lane, col]" in line, line
