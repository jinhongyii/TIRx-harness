"""v2 port of ``tests/numsim/integration/test_global_alias_artifact.py::test_global_decl_buffer_alias_reuses_parameter_allocation_bytes``.

Dropped pins:

- ``"extract_buffer_alias" in module.rust_source`` / ``"fn extract_buffer_alias("
  not in module.rust_source`` (legacy generated-Rust text);
- in-place mutation of the caller's ``storage`` array. Legacy ``Engine.run`` wrote
  the parameter bytes back into the numpy array passed in; v2 never mutates its
  inputs and returns every bound buffer in ``result.outputs`` (``v2/run.py``
  ``Engine.run``). The bytes are the same, so the assertion moves to
  ``result.outputs["storage"]``.

Kept: the alias written through the ``decl_buffer`` view reads back the source, and
the parameter allocation's bytes are the float32 bytes of ``source``.
"""

from __future__ import annotations

import numpy as np

from tests.numsim.support.kernels import global_decl_buffer_alias_reinterpret
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def test_global_decl_buffer_alias_reuses_parameter_allocation_bytes():
    """Port; see the module docstring for the dropped pins."""
    storage = np.zeros(128, dtype=np.uint8)
    source = np.linspace(-3, 5, 32, dtype=np.float32)
    output = np.zeros(32, dtype=np.float32)

    module = v2.transpile(global_decl_buffer_alias_reinterpret)
    result = v2.Engine().run(module, {"storage": storage, "source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)
    np.testing.assert_array_equal(result.outputs["storage"], source.view(np.uint8))
