"""v2 delta copy of ``tests/numsim/runtime/test_cvt_carriers.py::test_cvt_carriers_truncate_extend_and_gate_reads``
(public-API ``other-assertion`` triage, W11).

- Signed ``cvt`` into a wider carrier (e.g. ``cvt.s8.s8`` into ``int16``/``int32``)
  sign-extends; v2 used to resolve each PTX op with its first use's operand
  types (CONTRACT_REQUESTS "W11-other-assertion" W11-2, fixed by W2).
- Mode 1 (``pred=lane % 2 == 0`` without ``preserve_dst``): the odd (off) lanes
  keep the 0x5A fill instead of legacy's 0, numsim-behaviour-deltas P8 (TVM's
  write-only binding leaves the GPU value unspecified).

Kept: truncation/extension per carrier, f16/bf16 NaN canonicalisation, the
``pred=False`` and data-dependent predicate rows, and clean synccheck/racecheck.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.backend.cuda.ptx.table import TABLE, mods, operand_dtypes, tokens_for
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def cvt_carrier_case(ptx_type):
    entry = TABLE["cvt"]
    tokens = tokens_for(entry, dtype=ptx_type, atype=ptx_type)
    modifiers = mods(entry, tokens)
    destination_carriers = operand_dtypes(entry.operands[0], modifiers)
    source_carriers = operand_dtypes(entry.operands[1], modifiers)
    # Source truncation and destination extension are independent: vary each
    # against the canonical carrier instead of multiplying their combinations.
    forms = [(dst, source_carriers[0]) for dst in destination_carriers]
    forms += [(destination_carriers[0], src) for src in source_carriers[1:]]
    bits = tvm.DataType(source_carriers[0]).bits
    patterns = [
        0,
        1,
        (1 << bits) - 1,
        1 << (bits - 1),
        (1 << (bits - 1)) - 1,
        0x7C01,
        0xFC81,
        0x7F81,
        0xFF81,
        0x7F800001,
        0xFFC01234,
        0x7FF0000000000001,
        0xFFF81234567890AB,
    ]
    patterns += [i * 0x9E3779B97F4A7C15 for i in range(13, 32)]
    # Nonzero high halves distinguish source truncation from numeric conversion.
    source = np.asarray(
        [
            list(((value | (0xABCDEF0123456789 << 64)) & ((1 << 128) - 1)).to_bytes(16, "little"))
            for value in patterns
        ],
        np.uint8,
    )
    expected = np.full((4 * len(forms), 32, 16), 0x5A, np.uint8)
    lines = []
    for form_index, (destination_dtype, source_dtype) in enumerate(forms):
        dst, src = f"dst_{form_index}", f"src_{form_index}"
        dbytes = tvm.DataType(destination_dtype).bits // 8
        sbytes = tvm.DataType(source_dtype).bits // 8
        lines += [
            f'    storage_{form_index} = T.alloc_local((1,), "{destination_dtype}")',
            f'    {dst} = storage_{form_index}.view("{destination_dtype}")',
            f'    {src} = T.alloc_local((1,), "{source_dtype}")',
            f'    dst_bytes_{form_index} = {dst}.view("uint8")',
            f'    src_bytes_{form_index} = {src}.view("uint8")',
        ]
        # Inputs remain uninitialized on odd lanes, which the predicate must not read.
        lines += [
            "    if lane % 2 == 0:",
            f"        for byte in T.unroll({sbytes}):",
            f"            src_bytes_{form_index}[byte] = source[lane, byte]",
        ]
        for mode, predicate in (
            (0, ", pred=lane % 2 == 0, preserve_dst=True"),
            (1, ", pred=lane % 2 == 0"),
            (2, ", pred=False, preserve_dst=True"),
            (3, f", pred=dst_bytes_{form_index}[0] != T.uint8(0), preserve_dst=True"),
        ):
            row = 4 * form_index + mode
            lines += [
                f"    for byte in T.unroll({dbytes}):",
                f"        dst_bytes_{form_index}[byte] = T.uint8(0x5A)",
            ]
            if mode == 3:
                lines += [
                    "    if lane % 2 != 0:",
                    f"        dst_bytes_{form_index}[0] = T.uint8(0)",
                ]
                expected[row, 1::2, 0] = 0
            lines += [
                f'    T.ptx["cvt.{ptx_type}.{ptx_type}"]({dst}[0], {src}[0]{predicate})',
                f"    for byte in T.unroll({dbytes}):",
                f"        output[{row}, lane, byte] = dst_bytes_{form_index}[byte]",
            ]
            for lane in range(32):
                if lane % 2 == 0 and mode != 2:
                    value = int.from_bytes(
                        bytes(source[lane, : bits // 8]), "little", signed=ptx_type.startswith("s")
                    )
                    if ptx_type in {"f16", "bf16"}:
                        # B200 same-type CVT canonicalizes every half NaN;
                        # unlike MOV it is not an unconditional bitwise identity.
                        infinity = 0x7C00 if ptx_type == "f16" else 0x7F80
                        if value & 0x7FFF > infinity:
                            value = 0x7FFF
                    value &= (1 << (8 * dbytes)) - 1
                    expected[row, lane, :dbytes] = list(value.to_bytes(dbytes, "little"))
                # mode 1 (guarded, no preserve_dst) off lanes keep 0x5A: delta P8
                # (legacy wrote 0; the GPU value is unspecified).
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(source: T.Buffer((32, 16), "uint8"), output: T.Buffer(({len(expected)}, 32, 16), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
{chr(10).join(lines)}
""",
        {"T": T},
    )
    return kernel, dict(source=source, output=np.full_like(expected, 0x5A)), expected


def _types(kind):
    widths = (8, 16, 32, 64) if kind != "f" else (16, 32, 64)
    return [f"{kind}{width}" for width in widths] + (["bf16"] if kind == "f" else [])


def _types(kind):
    widths = (8, 16, 32, 64) if kind != "f" else (16, 32, 64)
    return [f"{kind}{width}" for width in widths] + (["bf16"] if kind == "f" else [])


@pytest.mark.parametrize("kind", ("u", "s", "f"))
@v2_gap("W11-4: 128-bit destination carriers (int128/uint128) of a narrow cvt stop as unsupported")
def test_cvt_carriers_truncate_extend_and_gate_reads(kind):
    """Delta copy; see the module docstring (P8 for mode 1 off lanes)."""
    for ptx_type in _types(kind):
        kernel, inputs, expected = cvt_carrier_case(ptx_type)
        for checker in (v2.synccheck, v2.racecheck):
            checker(kernel, dict(inputs)).require_clean()
        result = v2.Engine().run(v2.transpile(kernel), dict(inputs))
        np.testing.assert_array_equal(result.outputs["output"], expected)
