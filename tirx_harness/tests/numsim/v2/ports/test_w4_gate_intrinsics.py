"""v2 port of ``tests/numsim/runtime/test_gate_intrinsics.py`` (W4).

The legacy test compared ``abs`` / ``log1p`` / ``sigmoid`` with numpy at
``rtol=atol=2e-6`` and pinned the generated Rust text (``.abs()``,
``.ln_1p()``). The port keeps the numeric comparison, drops the text pins,
and replaces them with what they stood for, checked on the outputs:

* ``abs`` is exact (bit equality);
* ``log1p`` is within 1 binary32 ulp, and ``sigmoid`` within 2, of an
  independent binary64 reference rounded once;
* the special values of the host rule (numsim behaviour delta D9) give exact
  bits: NaN quieted (sigmoid flips its sign), ``x < -1`` gives the default NaN
  ``0xffc00000``, ``-1`` gives ``-inf``, and ``±0`` / ``+inf`` pass through.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def gate_intrinsics(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32, 3), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.float32 = source[lane]
    output[lane, 0] = T.abs(value)
    output[lane, 1] = T.log1p(value)
    output[lane, 2] = T.sigmoid(value)


def _ulps(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    def key(v):
        bits = v.astype(np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.int64(-(2**31)) - bits, bits)

    return np.abs(key(actual) - key(expected))


def _run(source: np.ndarray) -> np.ndarray:
    output = np.zeros((32, 3), dtype=np.float32)
    result = v2.Engine().run(v2.transpile(gate_intrinsics), {"source": source, "output": output})
    return result.outputs["output"]


def test_gate_intrinsics_match_float32_semantics():
    """Port of ``test_gate_intrinsics_match_float32_semantics`` (rust_source pins dropped)."""

    source = np.linspace(-0.875, 8.0, 32, dtype=np.float32)
    actual = _run(source)
    expected = np.stack(
        (
            np.abs(source),
            np.log1p(source),
            np.float32(1) / (np.float32(1) + np.exp(-source)),
        ),
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual[:, 0].view(np.uint32), np.abs(source).view(np.uint32))
    wide = source.astype(np.float64)
    assert _ulps(actual[:, 1], np.log1p(wide).astype(np.float32)).max() <= 1
    assert _ulps(actual[:, 2], (1.0 / (1.0 + np.exp(-wide))).astype(np.float32)).max() <= 2


def test_gate_intrinsics_special_values_follow_the_host_rule():
    """Replaces the ``.abs()`` / ``.ln_1p()`` text pins with observable bits (delta D9)."""

    specials = np.array(
        [0x7FC01234, 0xBF800000, 0xC0000000, 0xFF800000, 0x80000000, 0x00000000, 0x7F800000, 0x00000001],
        dtype=np.uint32,
    )
    source = np.zeros(32, dtype=np.uint32)
    source[: specials.size] = specials
    actual = _run(source.view(np.float32)).view(np.uint32)
    log1p = actual[: specials.size, 1]
    np.testing.assert_array_equal(
        log1p,
        np.array(
            [0x7FC01234, 0xFF800000, 0xFFC00000, 0xFFC00000, 0x80000000, 0x00000000, 0x7F800000, 0x00000001],
            dtype=np.uint32,
        ),
    )
    sigmoid = actual[: specials.size, 2]
    assert sigmoid[0] == 0xFFC01234  # NaN: sign flipped by the formula's -x, quieted
    assert sigmoid[5] == np.float32(0.5).view(np.uint32)
    assert sigmoid[6] == np.float32(1.0).view(np.uint32)
    assert sigmoid[3] == 0  # sigmoid(-inf) = 1 / (1 + inf) = +0
    np.testing.assert_array_equal(actual[: specials.size, 0], specials & np.uint32(0x7FFFFFFF))
