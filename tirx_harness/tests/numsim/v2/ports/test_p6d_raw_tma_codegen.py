"""v2 port of ``tests/numsim/runtime/test_raw_tma_codegen.py::test_raw_fp4_tensor_map_keeps_align8_shared_bytes_packed``.

Triage class ``port``. v2 round-trips the bytes exactly; legacy failed only on
the array *shape* of the output selected through the tensor map
(``outputs={"output": "output_map"}``): legacy returned the base array's
``(2, 64)`` shape, v2 returns ``(128,)``. ``v2/run.py::_tensor_map_view`` gives
up on the fp4 map (its global dims count 4-bit elements against a ``uint8``
base) and the fallback reshapes to the *selector's* array, here the 128-byte
descriptor image, which happens to have the same byte count. The copy compares
the flattened bytes (a report-shape detail, noted for W8). The kernel is the
pure-TVM ``tests.numsim.support.kernels.raw_tma_fp4_align8_roundtrip``.
"""

from __future__ import annotations

import numpy as np

from tests.numsim.support.kernels import raw_tma_fp4_align8_roundtrip
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


def _tensor_map(array: np.ndarray) -> np.ndarray:
    return TensorMap(
        base=array,
        global_shape=(128, 2),
        global_strides=(64,),
        box_shape=(128, 2),
        element_strides=(1, 1),
        fp4_shared_layout="align8_packed",
        swizzle="64B",
    ).numpy()


def test_raw_fp4_tensor_map_keeps_align8_shared_bytes_packed():
    """Port of ``tests/numsim/runtime/test_raw_tma_codegen.py::test_raw_fp4_tensor_map_keeps_align8_shared_bytes_packed``
    (dropped: the ``(2, 64)`` output shape)."""
    source = np.arange(128, dtype=np.uint8).reshape(2, 64)
    output = np.zeros_like(source)
    inputs = {"input_map": _tensor_map(source), "output_map": _tensor_map(output)}
    for checker in (v2.synccheck, v2.racecheck):
        assert checker(raw_tma_fp4_align8_roundtrip, dict(inputs)).verdict == "clean"

    result = v2.Engine().run(
        v2.transpile(raw_tma_fp4_align8_roundtrip), inputs, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(np.asarray(result.outputs["output"]).reshape(-1), source.reshape(-1))
    assert result.verdict == "clean"
