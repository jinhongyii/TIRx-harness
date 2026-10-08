"""v2 copy of ``tests/numsim/microtests/test_tcgen_inactive_boundaries.py::test_inactive_tcgen_boundary_matches_gpu``.

SM100 skips invalid MMA descriptors when their instruction predicate is false.
``BOUNDARY_CASES`` and ``inactive_boundary_case`` are copied verbatim from
``tests/numsim/runtime/test_tcgen_inactive_boundaries.py`` (removed at step 5)
with ``decode_ptx_call`` replaced by the TVM PTX-table operand decoder of the
p8 descriptor-dispatch port. GPU-only (``numsim_gpu``): the NumSim side runs
on v2; the GPU side uses the legacy-free ``microtests/harness``.
"""

import numpy as np
import pytest
from tvm import tirx
from tvm.script import tirx as T
from tvm_ffi import structural_map

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tests.numsim.v2.checkers._tcgen_kernels import ti16_kernel
from tests.numsim.v2.ports.test_p8_tcgen_descriptor_dispatch import _ptx_operands, _scalar
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


# Distinct instruction dispatch paths, not a Cartesian modifier matrix.
BOUNDARY_CASES = (
    ("mixed_ss", "f16", False, False, False, 0),
    ("mixed_ts", "f16", True, False, False, 0),
    ("mixed_ws", "f16", False, True, False, 0),
    ("mixed_sparse", "f16", False, False, True, 0),
    ("tf32_ts", "tf32", True, False, False, 15),
    ("i8_ts", "i8", True, False, False, 15),
    ("f8_ts", "f8f6f4", True, False, False, 15),
    ("f16_sparse_ts", "f16", True, False, True, 15),
    ("ti16_transpose_a", "ti16", False, False, False, 15),
    ("ti16_transpose_b", "ti16", False, False, False, 16),
)


def inactive_boundary_case(case):
    name, kind, tmem, ws, sparse, transpose_bit = case
    kernel = ti16_kernel(
        True,
        tmem,
        ws=ws,
        kind=kind,
        sparse=sparse,
        b_format=int(name.startswith("mixed")),
        collectors=".collector::a::discard.collector::b::discard" if sparse else "",
        arch="sm_107a" if sparse or kind == "ti16" else "sm_100a",
    )
    if transpose_bit:

        def set_transpose(node):
            call = node.value
            if type(call).__name__ != "Call" or not str(call.op.name).startswith(
                "tirx.ptx.tcgen05_mma"
            ):
                return node
            descriptor = _scalar(_ptx_operands(call), "idesc")
            args = [
                T.uint32(int(arg.value) | (1 << transpose_bit)) if arg.same_as(descriptor) else arg
                for arg in call.args
            ]
            return tirx.Evaluate(
                type(call)(
                    call.op,
                    args,
                    attrs=call.attrs,
                    ty_args=call.ty_args,
                    span=call.span,
                    ret_ty=call.ty,
                )
            )

        kernel = kernel.with_body(structural_map(kernel.body, (tirx.Evaluate, set_transpose)))
    columns = 64 if ws else 16
    inputs = {
        "a": np.zeros((4, 128, 16), np.uint16),
        "b": np.zeros((96 if ws else 16, 32 if sparse else 16), np.uint16),
        "metadata": np.zeros((2, 128, 2), np.uint32),
        "seed": np.zeros((128, columns), np.int32),
        "out": np.zeros((128, columns), np.int32),
        "zero_mask": np.zeros(1, np.uint64),
    }
    inputs["seed"][:] = np.arange(inputs["seed"].size, dtype=np.int32).reshape(inputs["seed"].shape)
    inputs["zero_mask"][0] = np.uint64(1 << 63)
    reason = (
        "requires matching F16/BF16"
        if name.startswith("mixed")
        else "ti16_transpose_unmodeled"
        if kind == "ti16"
        else "TMEM A must be K-major"
    )
    return kernel, inputs, reason


SM100_CASES = tuple(case for case in BOUNDARY_CASES if case[1] != "ti16" and not case[4])


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("case", SM100_CASES, ids=[case[0] for case in SM100_CASES])
def test_inactive_tcgen_boundary_matches_gpu(pytestconfig, case):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, _ = inactive_boundary_case(case)
    cpu = v2.Engine().run(v2.transpile(kernel), dict(inputs), outputs=("out",))
    np.testing.assert_array_equal(cpu.outputs["out"], inputs["seed"])
    gpu = run_gpu_primfunc(kernel, inputs, outputs=("out",), arch="sm_100a")
    np.testing.assert_array_equal(gpu["out"], inputs["seed"])


@pytest.mark.parametrize("case", SM100_CASES, ids=[case[0] for case in SM100_CASES])
def test_inactive_tcgen_boundary_numsim_side(case):
    """The NumSim half of the GPU oracle, runnable without a GPU."""
    kernel, inputs, _ = inactive_boundary_case(case)
    cpu = v2.Engine().run(v2.transpile(kernel), dict(inputs), outputs=("out",))
    np.testing.assert_array_equal(cpu.outputs["out"], inputs["seed"])
