"""v2 copies of two ``tests/numsim/integration/test_scalar_parameter_artifact.py``
functions (W9 phase 6, internal other-assertion triage). Both **port**.

- ``test_plain_and_numpy_int32_scalar_parameters_affect_generated_code``:
  values unchanged. Dropped the legacy ``module.spec.kernels[0].scalars`` pin
  (v2 ``KernelSpec`` has no ``scalars``; the scalar is a ``Scalar`` slot of
  ``host_abi`` with dtype ``S32``, asserted instead) and the three
  ``rust_source`` pins.
- ``test_multi_kernel_scalar_names_are_phase_qualified``: values unchanged;
  the phase-qualified ``k0:``/``k1:`` input names still bind. Dropped the
  default output key names: with no ``outputs=`` v2 names each output by its
  canonical buffer name (``output``, ``second``), legacy by ``k0:output`` /
  ``k1:second``. The copy asserts both forms: default keys and explicit
  ``outputs=("k0:output", "k1:second")`` selectors. No delta row documents
  the default output naming (W8 may want one).
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def scalar_parameter_add(offset: T.int32, output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = offset + lane


@T.prim_func
def second_scalar_parameter_add(offset: T.int32, second: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    second[lane] = offset - lane


def test_plain_and_numpy_int32_scalar_parameters_affect_generated_code():
    """Port of ``tests/numsim/integration/test_scalar_parameter_artifact.py::test_plain_and_numpy_int32_scalar_parameters_affect_generated_code``."""

    output = np.zeros(32, dtype=np.int32)
    module = v2.transpile(scalar_parameter_add)

    plain = v2.Engine().run(module, {"offset": 7, "output": output})
    np.testing.assert_array_equal(plain.outputs["output"], np.arange(32, dtype=np.int32) + 7)

    explicit = v2.Engine().run(module, {"offset": np.int32(-3), "output": output})
    np.testing.assert_array_equal(explicit.outputs["output"], np.arange(32, dtype=np.int32) - 3)

    scalars = [slot for slot in module.spec.kernels[0].host_abi if slot["kind"] == "Scalar"]
    assert [slot["name"] for slot in scalars] == ["offset"]
    assert scalars[0]["dtype"] == {"elem": "S32", "lanes": 1}


def test_multi_kernel_scalar_names_are_phase_qualified():
    """Port of ``tests/numsim/integration/test_scalar_parameter_artifact.py::test_multi_kernel_scalar_names_are_phase_qualified``."""

    module = v2.transpile([scalar_parameter_add, second_scalar_parameter_add])
    inputs = {
        "k0:offset": 2,
        "k0:output": np.zeros(32, dtype=np.int32),
        "k1:offset": 5,
        "k1:second": np.zeros(32, dtype=np.int32),
    }
    first = np.arange(32, dtype=np.int32) + 2
    second = 5 - np.arange(32, dtype=np.int32)

    result = v2.Engine().run(module, inputs)
    np.testing.assert_array_equal(result.outputs["output"], first)
    np.testing.assert_array_equal(result.outputs["second"], second)

    selected = v2.Engine().run(module, inputs, outputs=("k0:output", "k1:second"))
    np.testing.assert_array_equal(selected.outputs["k0:output"], first)
    np.testing.assert_array_equal(selected.outputs["k1:second"], second)
