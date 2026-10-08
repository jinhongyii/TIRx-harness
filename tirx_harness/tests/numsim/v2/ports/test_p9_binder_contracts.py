"""v2 copies of binder-contract tests from
``tests/numsim/integration/test_multi_kernel_artifact.py`` and
``tests/numsim/integration/test_scalar_parameter_artifact.py`` (W12).

Legacy's rejection is the oracle, except where numsim-behaviour-deltas H7
rules the documented v2 binder rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.support.multi_kernels import (
    ambiguous_alias_first,
    ambiguous_alias_second,
    consume_intermediate,
    write_intermediate,
)
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tests.numsim.v2.ports.test_p6b_scalar_parameter_artifact import scalar_parameter_add
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def test_multi_kernel_rejects_an_ambiguous_output_alias():
    """Copy of ``test_multi_kernel_artifact.py::test_multi_kernel_rejects_an_ambiguous_output_alias``."""

    module = v2.transpile([ambiguous_alias_first, ambiguous_alias_second])
    with pytest.raises(v2.InputError, match="output alias 'shared' is ambiguous"):
        v2.Engine().run(
            module,
            {"k0:shared": np.zeros(32, np.float32), "k1:shared": np.zeros(32, np.float32)},
            outputs=("shared",),
        )
    # A qualified selector is unambiguous.
    result = v2.Engine().run(
        module,
        {"k0:shared": np.zeros(32, np.float32), "k1:shared": np.zeros(32, np.float32)},
        outputs=("k1:shared",),
    )
    assert set(result.outputs) == {"k1:shared"}


def test_multi_kernel_duplicate_qualified_aliases_are_one_binding_when_equal():
    """Delta H7 of ``test_multi_kernel_artifact.py::test_multi_kernel_rejects_duplicate_qualified_aliases``.

    Legacy rejected ``k1:output`` and ``output`` naming the same parameter.
    v2 binds the same object (or equal bytes) under several names once, and
    rejects different values with a typed ``InputError``."""

    module = v2.transpile([write_intermediate, consume_intermediate])
    intermediate = np.zeros(32, dtype=np.float32)
    output = np.zeros((2, 32), dtype=np.float32)
    inputs = {"k0:intermediate": intermediate, "k1:intermediate": intermediate,
              "k1:output": output, "output": output}
    result = v2.Engine().run(module, inputs)
    assert "k1:output" in result.outputs or "output" in result.outputs

    with pytest.raises(v2.InputError, match="bound more than once with different values"):
        v2.Engine().run(module, {**inputs, "output": np.ones((2, 32), dtype=np.float32)})


def test_generated_artifact_scalar_dtype_accepts_fitting_integers():
    """Delta H8 of ``test_scalar_parameter_artifact.py::test_generated_artifact_rejects_wrong_scalar_dtype``,
    and the range half of ``test_scalar_parameter_range_is_checked_before_execution``.

    Legacy rejected a NumPy integer of another width (``np.uint32(7)`` for an
    int32 parameter); v2 accepts any integer value that fits and rejects
    out-of-range values and non-integers with a typed ``InputError``."""

    module = v2.transpile(scalar_parameter_add)
    output = np.zeros(32, dtype=np.int32)
    result = v2.Engine().run(module, {"offset": np.uint32(7), "output": output})
    np.testing.assert_array_equal(result.outputs["output"], np.arange(32, dtype=np.int32) + 7)
    with pytest.raises(v2.InputError, match="outside int32 range"):
        v2.Engine().run(module, {"offset": 1 << 31, "output": output})
    with pytest.raises(v2.InputError, match="scalar 'offset' requires dtype int32, got float32"):
        v2.Engine().run(module, {"offset": np.float32(7), "output": output})
