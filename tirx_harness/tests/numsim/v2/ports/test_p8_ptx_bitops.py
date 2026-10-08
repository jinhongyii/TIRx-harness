"""v2 copies of the ``tests/numsim/runtime/test_ptx_bitops.py`` functions that
do not use the schema-derived form list but break at import once the legacy
``numsim.transpiler.ptx_dialect.PTX_SCHEMA_BY_OP_NAME`` (module level) is
deleted. The schema-driven all-forms test is already ported in
``test_p6d_ptx_bitops.py``.

Kernels, inputs and oracles are copied verbatim. Changes:

- ``run_checked`` (legacy synccheck/racecheck ``require_clean`` + legacy
  ``numsim.Engine``) becomes v2 ``synccheck``/``racecheck`` clean verdicts and
  ``v2.Engine().run``.
- ``NumSimExecutionError`` becomes ``v2.ExecutionError``; the legacy
  ``match=`` text is kept as a match on the v2 error message.
- ``cache_dir=tmp_path`` is dropped (v2 has no on-disk build cache).
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import assert_clean, requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def _run_checked(kernel, inputs, *, outputs=None):
    for checker in (v2.synccheck, v2.racecheck):
        assert_clean(checker(kernel, {name: np.copy(value) for name, value in inputs.items()}))
    return v2.Engine().run(v2.transpile(kernel), inputs, outputs=outputs)


@pytest.mark.parametrize(
    "spelling", ("bfe.u32", "bfe.s32", "bfe.u64", "bfe.s64", "bfi.b32", "bfi.b64")
)
def test_bitfield_control_domain_is_checked_only_on_issuing_lanes(spelling):
    """Port of ``tests/numsim/runtime/test_ptx_bitops.py::test_bitfield_control_domain_is_checked_only_on_issuing_lanes``.

    The legacy loop over the six spellings becomes a parametrization.
    """

    bits = int(spelling[-2:])
    dtype = f"{'int' if spelling.startswith('bfe.s') else 'uint'}{bits}"
    data = f"T.{dtype}(1), " + (f"T.{dtype}(0), " if spelling.startswith("bfi") else "")
    kernel = tvm.script.from_source(
        f'''@T.prim_func
def kernel(position: T.Buffer((32,), "uint32"), length: T.Buffer((32,), "uint32"),
           output: T.Buffer((32,), "{dtype}")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.{dtype}(85)
    T.ptx["{spelling}"](output[lane], {data}position[lane], length[lane],
                      pred=lane % 2 == 0, preserve_dst=True)
''',
        {"T": T},
    )
    inputs = dict(
        position=np.zeros(32, np.uint32),
        length=np.ones(32, np.uint32),
        output=np.zeros(32, dtype),
    )
    inputs["position"][1::2] = 0xFFFFFFFF
    inputs["length"][1::2] = 0xFFFFFFFF
    result = _run_checked(kernel, inputs)
    # A signed one-bit field containing 1 sign-extends to -1.
    extracted = -1 if spelling.startswith("bfe.s") else 1
    np.testing.assert_array_equal(result.outputs["output"], [extracted, 85] * 16)
    module = v2.transpile(kernel)
    for control in ("position", "length"):
        for invalid in (256, 0xFFFFFFFF):
            original = inputs[control][0]
            inputs[control][0] = invalid
            with pytest.raises(v2.ExecutionError, match="outside the defined 0..255 range"):
                v2.Engine().run(module, inputs)
            inputs[control][0] = original


@T.prim_func
def _fns_kernel(bases: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.uint32(85)
    T.ptx.fns.b32(
        output[lane],
        T.uint32(0xFFFFFFFF),
        bases[lane],
        T.int32(-1),
        pred=lane % 2 == 0,
        preserve_dst=True,
    )


def test_fns_base_domain_is_checked_only_on_issuing_lanes():
    """Port of ``tests/numsim/runtime/test_ptx_bitops.py::test_fns_base_domain_is_checked_only_on_issuing_lanes``."""

    kernel = _fns_kernel
    inputs = dict(bases=np.arange(32, dtype=np.uint32), output=np.zeros(32, np.uint32))
    inputs["bases"][1::2] = 0xFFFFFFFF
    for checker in (v2.synccheck, v2.racecheck):
        assert_clean(checker(kernel, {name: np.copy(value) for name, value in inputs.items()}))
    module = v2.transpile(kernel)
    expected = np.arange(32, dtype=np.uint32)
    expected[1::2] = 85
    np.testing.assert_array_equal(v2.Engine().run(module, inputs).outputs["output"], expected)
    for invalid in (32, 0xFFFFFFFF):
        inputs["bases"][0] = invalid
        with pytest.raises(
            v2.ExecutionError, match="fns.b32 base is outside the defined 0..31 range"
        ):
            v2.Engine().run(module, inputs)


_B16_DTYPES = ("uint16", "int16", "float16", "bfloat16")


def _make_b16_carrier_kernel():
    lines = [
        "@T.prim_func",
        "def ptx_b16_carriers(",
        *(f'    input_{dtype}: T.Buffer((32,), "{dtype}"),' for dtype in _B16_DTYPES),
        *(f'    output_{dtype}: T.Buffer((8, 32), "{dtype}"),' for dtype in _B16_DTYPES),
        "):",
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
    ]
    for destination_dtype in _B16_DTYPES:
        row = 0
        for instruction in ("not", "cnot"):
            for source_dtype in _B16_DTYPES:
                lines.append(
                    f'    T.ptx["{instruction}.b16"]('
                    f"output_{destination_dtype}[{row}, lane], input_{source_dtype}[lane])"
                )
                row += 1
    return tvm.script.from_source("\n".join(lines), {"T": T})


def _view_b16(bits: np.ndarray, dtype: str) -> np.ndarray:
    return bits.view(
        {
            "uint16": np.uint16,
            "int16": np.int16,
            "float16": np.float16,
            "bfloat16": ml_dtypes.bfloat16,
        }[dtype]
    )


def test_b16_carrier_cross_product_preserves_every_payload_bit():
    """Port of ``tests/numsim/runtime/test_ptx_bitops.py::test_b16_carrier_cross_product_preserves_every_payload_bit``."""

    edge_bits = np.asarray(
        [
            0x0000,
            0x0001,
            0x03FF,
            0x0400,
            0x3C00,
            0x7BFF,
            0x7C00,
            0x7C01,
            0x7DFF,
            0x7E00,
            0x7FFF,
            0x8000,
            0xFC00,
            0xFC01,
            0xFE00,
            0xFFFF,
            *((0x9E37 * lane + 0x1234) & 0xFFFF for lane in range(16, 32)),
        ],
        dtype=np.uint16,
    )
    source_bits = {
        dtype: np.bitwise_xor(edge_bits, np.uint16(index * 0x1111))
        for index, dtype in enumerate(_B16_DTYPES)
    }
    arguments = {
        **{f"input_{dtype}": _view_b16(bits, dtype) for dtype, bits in source_bits.items()},
        **{
            f"output_{dtype}": _view_b16(
                np.full((8, 32), np.uint16(0xDEAD), dtype=np.uint16), dtype
            )
            for dtype in _B16_DTYPES
        },
    }
    result = v2.Engine().run(
        v2.transpile(_make_b16_carrier_kernel()),
        arguments,
        outputs=tuple(f"output_{dtype}" for dtype in _B16_DTYPES),
    )
    expected = np.stack(
        [
            *(np.bitwise_not(source_bits[dtype]) for dtype in _B16_DTYPES),
            *((source_bits[dtype] == 0).astype(np.uint16) for dtype in _B16_DTYPES),
        ]
    )
    for dtype in _B16_DTYPES:
        np.testing.assert_array_equal(result.outputs[f"output_{dtype}"].view(np.uint16), expected)
