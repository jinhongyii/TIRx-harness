"""v2 ports of the ``uses-legacy-internals`` tests in
``tests/numsim/registry/test_call_resolution.py``.

The legacy tests read the legacy frontend spec (``analyze(...).tensor_maps``
as ``TensorMapSpec`` tuples) and the legacy ``build_host_abi``. The ports
read the same facts from the v2 module's host ABI (``CompiledModule.spec``)
and keep the fail-closed contract through ``v2.transpile``/``v2.Engine``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm import tirx
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tests.numsim.support.kernels import (
    no_op_kernel,
    raw_tma_roundtrip,
    tcgen_lifecycle_single_cta,
    tcgen_tmem_to_local_roundtrip,
)
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


def _public_tensormap_prefetch(parameter_name: str):
    func = tvm.script.from_source(
        f"""
@T.prim_func
def kernel({parameter_name}: T.TensorMap()):
    T.device_entry()
    T.evaluate(T.ptx.prefetch.tensormap(T.address_of({parameter_name})))
""",
        {"T": T},
    )
    calls = []

    def visit(node: object) -> None:
        if type(node).__name__ == "Call" and str(getattr(node.op, "name", "")) == "tirx.ptx.prefetch":
            calls.append(node)

    structural_walk(func.body, ((Expr, Stmt), visit))
    assert len(calls) == 1
    return func, calls[0]


def _tensor_maps(module: v2.CompiledModule) -> list[tuple[str, int]]:
    return [
        (entry["name"], entry["buf"])
        for entry in module.spec.kernels[0].host_abi
        if entry["kind"] == "TensorMap"
    ]


def test_frontend_discovers_typed_tensor_map_parameters_from_exact_owners():
    """Port of ``tests/numsim/registry/test_call_resolution.py::test_frontend_discovers_typed_tensor_map_parameters_from_exact_owners``.

    ``TensorMapSpec("input_map", 0), TensorMapSpec("output_map", 1)`` becomes
    the v2 host ABI's two ``TensorMap`` entries in parameter order. Dropped
    pin: ``"tirx.address_of" in call_op_names(spec)`` (legacy IR walk).
    """

    module = v2.transpile(raw_tma_roundtrip)
    assert _tensor_maps(module) == [("input_map", 0), ("output_map", 1)]


@v2_gap(
    "v2 accepts two distinct TensorMap parameters that share the public name "
    "'descriptor' (host ABI lists both, run completes); legacy build_host_abi "
    "raised HostAbiError 'host binding \"descriptor\"' for the ambiguous binding"
)
def test_host_abi_rejects_distinct_tensor_map_parameters_with_one_public_name():
    """Port of ``tests/numsim/registry/test_call_resolution.py::test_host_abi_rejects_distinct_tensor_map_parameters_with_one_public_name``.

    The two parameters are discovered in order (as in legacy); the host
    binding by name is ambiguous and must fail closed at transpile or at
    binding time. Dropped: the legacy ``HostAbiError`` type and message.
    """

    first_func, first_prefetch = _public_tensormap_prefetch("descriptor")
    second_func, second_prefetch = _public_tensormap_prefetch("descriptor")
    body = tirx.SeqStmt([tirx.Evaluate(first_prefetch), tirx.Evaluate(second_prefetch)])
    func = tirx.PrimFunc([first_func.params[0], second_func.params[0]], body)

    with pytest.raises((UnsupportedTIRxError, v2.ModuleContractError, v2.InputError)):
        module = v2.transpile(func)
        assert _tensor_maps(module) == [("descriptor", 0), ("descriptor", 1)]
        v2.Engine().run(module, {})


def test_frontend_records_typed_implicit_tmem_requirement():
    """Port of ``tests/numsim/registry/test_call_resolution.py::test_frontend_records_typed_implicit_tmem_requirement``.

    The legacy ``requires_implicit_tmem`` flag was true for the kernel that
    allocates TMEM through ``tcgen05.alloc`` and false for the static
    ``decl_buffer(scope="tmem", allocated_addr=0)`` view. The v2 module
    carries the opposite definition under ``requirements.implicit_tmem``
    (true only for static-address TMEM views with no ``tcgen05.alloc``; the
    alloc kernel gets its TMEM from the lease). The port asserts the v2
    definition, and that all three kernels run to completion, which is the
    observable requirement the flag records.
    """

    runs = (
        (tcgen_lifecycle_single_cta, {"output": np.zeros(1, np.uint32)}, False),
        (
            tcgen_tmem_to_local_roundtrip,
            {
                "source": np.arange(512, dtype=np.float32).reshape(128, 4),
                "output": np.zeros((128, 4), np.float32),
            },
            True,
        ),
        (no_op_kernel, {}, False),
    )
    for kernel, inputs, implicit in runs:
        module = v2.transpile(kernel)
        assert module.document["kernels"][0]["requirements"]["implicit_tmem"] is implicit, kernel
        assert v2.Engine().run(module, inputs).status["kind"] == "completed"
