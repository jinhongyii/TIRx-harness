"""v2 ports of ``tests/numsim/integration/test_multi_kernel_artifact.py`` tests
that pinned ``module.rust_source`` or legacy binding-error text. Kernels are
copied verbatim from ``tests/numsim/support/multi_kernels.py``. Binding errors
are asserted as ``v2.InputError`` (not ``MissingBindingsError``), with a
positive control showing the phase-qualified spelling of the same inputs runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def ambiguous_alias_first(shared: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared[lane] = T.float32(1)


@T.prim_func
def ambiguous_alias_second(shared: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared[lane] = T.float32(2)


@T.prim_func
def same_names_first(
    input: T.Buffer((32,), "float32"),
    intermediate: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
    scale: T.int32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    intermediate[lane] = input[lane] * T.cast(scale, "float32")
    output[lane] = input[lane] + T.float32(1)


@T.prim_func
def same_names_second(
    input: T.Buffer((32,), "float32"),
    intermediate: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
    scale: T.float32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = input[lane] + intermediate[lane] * scale


@T.prim_func
def typed_pointer_first(pointer: T.handle("uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.ptr_byte_offset(pointer, T.cast(lane * 4, "uint32"), "uint32")
    T.ptx.ld.global_.u32(output[lane], address)


@T.prim_func
def typed_pointer_second(pointer: T.handle("uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.ptr_byte_offset(pointer, T.cast(lane * 4, "uint32"), "uint32")
    loaded = T.local_scalar("uint32")
    T.ptx.ld.global_.u32(loaded, address)
    output[lane] = loaded + T.uint32(1)


def _same_name_inputs(*, shared_intermediate: bool):
    input0 = np.arange(32, dtype=np.float32)
    input1 = np.arange(32, dtype=np.float32) + np.float32(100)
    intermediate0 = np.zeros(32, dtype=np.float32)
    intermediate1 = intermediate0 if shared_intermediate else np.full(32, 7, dtype=np.float32)
    output0 = np.zeros(32, dtype=np.float32)
    output1 = np.zeros(32, dtype=np.float32)
    return {
        "k0:input": input0,
        "k0:intermediate": intermediate0,
        "k0:output": output0,
        "k0:scale": 3,
        "k1:input": input1,
        "k1:intermediate": intermediate1,
        "k1:output": output1,
        "k1:scale": np.float32(0.5),
    }


def _assert_binding_error(module, inputs) -> None:
    with pytest.raises(v2.InputError) as caught:
        v2.Engine().run(module, inputs)
    assert not isinstance(caught.value, v2.MissingBindingsError), caught.value


def test_multi_kernel_rejects_an_alias_with_multiple_canonical_targets():
    """Port of ``tests/numsim/integration/test_multi_kernel_artifact.py::test_multi_kernel_rejects_an_alias_with_multiple_canonical_targets``.

    Dropped: ``match="input alias 'shared' is ambiguous"`` (legacy text).
    Asserted: unqualified ``shared`` (a parameter of both kernels) raises
    ``v2.InputError``; the qualified ``k0:shared`` / ``k1:shared`` run.
    """

    module = v2.transpile([ambiguous_alias_first, ambiguous_alias_second])

    _assert_binding_error(module, {"shared": np.zeros(32, dtype=np.float32)})

    result = v2.Engine().run(
        module,
        {"k0:shared": np.zeros(32, dtype=np.float32), "k1:shared": np.zeros(32, dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["k0:shared"], np.ones(32, dtype=np.float32))
    np.testing.assert_array_equal(result.outputs["k1:shared"], np.full(32, 2, dtype=np.float32))


def test_multi_kernel_rejects_unqualified_scalar_collisions_and_unknown_inputs():
    """Port of ``tests/numsim/integration/test_multi_kernel_artifact.py::test_multi_kernel_rejects_unqualified_scalar_collisions_and_unknown_inputs``.

    Dropped: ``match="input alias 'scale' is ambiguous"`` and
    ``match="binding 'scsale' is unknown"`` (legacy text). Asserted: both
    input sets raise ``v2.InputError``, and the unmodified qualified inputs
    run.
    """

    module = v2.transpile([same_names_first, same_names_second])
    inputs = _same_name_inputs(shared_intermediate=False)
    inputs["scale"] = inputs.pop("k0:scale")
    _assert_binding_error(module, inputs)

    inputs = _same_name_inputs(shared_intermediate=False)
    inputs["scsale"] = 4
    _assert_binding_error(module, inputs)

    result = v2.Engine().run(module, _same_name_inputs(shared_intermediate=False))
    assert result.status.get("kind") == "completed", result.status


def test_multi_kernel_pointer_parameters_are_phase_qualified():
    """Port of ``tests/numsim/integration/test_multi_kernel_artifact.py::test_multi_kernel_pointer_parameters_are_phase_qualified``.

    Dropped pins: ``'"k0:pointer"'`` / ``'"k1:pointer"'`` in
    ``module.rust_source``. The per-phase outputs are asserted.
    """

    module = v2.transpile([typed_pointer_first, typed_pointer_second])
    source0 = np.arange(32, dtype=np.uint32)
    source1 = np.arange(32, dtype=np.uint32) + np.uint32(100)

    result = v2.Engine().run(
        module,
        {
            "k0:pointer": source0,
            "k0:output": np.zeros(32, dtype=np.uint32),
            "k1:pointer": source1,
            "k1:output": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["k0:output"], source0)
    np.testing.assert_array_equal(result.outputs["k1:output"], source1 + np.uint32(1))
