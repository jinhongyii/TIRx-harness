"""v2 copies of ``tests/numsim/runtime/test_ptx_float_register_ops.py``.

The legacy module builds its form list at import time from the legacy
``numsim.transpiler.ptx_dialect.PTX_SCHEMA_BY_OP_NAME``, so it breaks once the
legacy transpiler is deleted. The copy freezes that list as a literal
``_FORMS`` tuple (same forms, spellings, modifiers, dtypes and output rows).
Kernel builder, inputs and oracles are copied verbatim; legacy
``numsim.Engine().run(numsim.transpile(..., cache_dir=...))`` becomes
``v2.Engine().run(v2.transpile(...))`` (v2 has no on-disk build cache).
Nothing else is dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

COPYSIGN = "tirx.ptx.copysign"
MAD_F = "tirx.ptx.mad_f"
NEG_HALF = "tirx.ptx.neg_half"
COS = "tirx.ptx.cos"
SIN = "tirx.ptx.sin"
SQRT = "tirx.ptx.sqrt"

_OPS = (COPYSIGN, NEG_HALF, MAD_F, COS, SIN, SQRT)


@dataclass(frozen=True)
class _RuntimeForm:
    op_name: str
    spelling: str
    modifiers: tuple[tuple[str, str], ...]
    dtypes: tuple[str, ...]
    output_row: int


# Frozen from the legacy ``PTX_SCHEMA_BY_OP_NAME`` walk (``variants`` x
# ``mods`` x ``canonical_dtypes`` over ``_OPS``, in that order, with per-
# destination-dtype output rows) evaluated once on 2026-10-08 before the
# legacy transpiler was deleted. The legacy schema entries were the TVM
# ``tvm.backend.cuda.ptx.table.TABLE`` entries keyed by ``op_name``.
_FORMS: tuple[_RuntimeForm, ...] = (
    _RuntimeForm(
        COPYSIGN,
        "copysign.f32",
        (("type", "f32"),),
        ("float32", "float32", "float32"),
        0,
    ),
    _RuntimeForm(
        COPYSIGN,
        "copysign.f64",
        (("type", "f64"),),
        ("float64", "float64", "float64"),
        0,
    ),
    _RuntimeForm(
        NEG_HALF,
        "neg.ftz.f16",
        (("ftz", "ftz"), ("type", "f16")),
        ("uint16", "uint16"),
        0,
    ),
    _RuntimeForm(
        NEG_HALF,
        "neg.ftz.f16x2",
        (("ftz", "ftz"), ("type", "f16x2")),
        ("uint32", "uint32"),
        0,
    ),
    _RuntimeForm(
        NEG_HALF,
        "neg.f16",
        (("ftz", ""), ("type", "f16")),
        ("uint16", "uint16"),
        1,
    ),
    _RuntimeForm(
        NEG_HALF,
        "neg.f16x2",
        (("ftz", ""), ("type", "f16x2")),
        ("uint32", "uint32"),
        1,
    ),
    _RuntimeForm(
        NEG_HALF,
        "neg.bf16",
        (("ftz", ""), ("type", "bf16")),
        ("uint16", "uint16"),
        2,
    ),
    _RuntimeForm(
        NEG_HALF,
        "neg.bf16x2",
        (("ftz", ""), ("type", "bf16x2")),
        ("uint32", "uint32"),
        2,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rn.ftz.sat.f32",
        (("rnd", "rn"), ("ftz", "ftz"), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        1,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rn.ftz.f32",
        (("rnd", "rn"), ("ftz", "ftz"), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        2,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rn.sat.f32",
        (("rnd", "rn"), ("ftz", ""), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        3,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rn.f32",
        (("rnd", "rn"), ("ftz", ""), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        4,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rn.f64",
        (("rnd", "rn"), ("ftz", ""), ("sat", ""), ("type", "f64")),
        ("float64", "float64", "float64", "float64"),
        1,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rz.ftz.sat.f32",
        (("rnd", "rz"), ("ftz", "ftz"), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        5,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rz.ftz.f32",
        (("rnd", "rz"), ("ftz", "ftz"), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        6,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rz.sat.f32",
        (("rnd", "rz"), ("ftz", ""), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        7,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rz.f32",
        (("rnd", "rz"), ("ftz", ""), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        8,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rz.f64",
        (("rnd", "rz"), ("ftz", ""), ("sat", ""), ("type", "f64")),
        ("float64", "float64", "float64", "float64"),
        2,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rm.ftz.sat.f32",
        (("rnd", "rm"), ("ftz", "ftz"), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        9,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rm.ftz.f32",
        (("rnd", "rm"), ("ftz", "ftz"), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        10,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rm.sat.f32",
        (("rnd", "rm"), ("ftz", ""), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        11,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rm.f32",
        (("rnd", "rm"), ("ftz", ""), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        12,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rm.f64",
        (("rnd", "rm"), ("ftz", ""), ("sat", ""), ("type", "f64")),
        ("float64", "float64", "float64", "float64"),
        3,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rp.ftz.sat.f32",
        (("rnd", "rp"), ("ftz", "ftz"), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        13,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rp.ftz.f32",
        (("rnd", "rp"), ("ftz", "ftz"), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        14,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rp.sat.f32",
        (("rnd", "rp"), ("ftz", ""), ("sat", "sat"), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        15,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rp.f32",
        (("rnd", "rp"), ("ftz", ""), ("sat", ""), ("type", "f32")),
        ("float32", "float32", "float32", "float32"),
        16,
    ),
    _RuntimeForm(
        MAD_F,
        "mad.rp.f64",
        (("rnd", "rp"), ("ftz", ""), ("sat", ""), ("type", "f64")),
        ("float64", "float64", "float64", "float64"),
        4,
    ),
    _RuntimeForm(
        COS,
        "cos.approx.ftz.f32",
        (("mode", "approx"), ("ftz", "ftz"), ("type", "f32")),
        ("float32", "float32"),
        17,
    ),
    _RuntimeForm(
        COS,
        "cos.approx.f32",
        (("mode", "approx"), ("ftz", ""), ("type", "f32")),
        ("float32", "float32"),
        18,
    ),
    _RuntimeForm(
        SIN,
        "sin.approx.ftz.f32",
        (("mode", "approx"), ("ftz", "ftz"), ("type", "f32")),
        ("float32", "float32"),
        19,
    ),
    _RuntimeForm(
        SIN,
        "sin.approx.f32",
        (("mode", "approx"), ("ftz", ""), ("type", "f32")),
        ("float32", "float32"),
        20,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.approx.ftz.f32",
        (("mode", "approx"), ("ftz", "ftz"), ("type", "f32")),
        ("float32", "float32"),
        21,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.approx.f32",
        (("mode", "approx"), ("ftz", ""), ("type", "f32")),
        ("float32", "float32"),
        22,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rn.ftz.f32",
        (("mode", "rn"), ("ftz", "ftz"), ("type", "f32")),
        ("float32", "float32"),
        23,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rn.f32",
        (("mode", "rn"), ("ftz", ""), ("type", "f32")),
        ("float32", "float32"),
        24,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rn.f64",
        (("mode", "rn"), ("ftz", ""), ("type", "f64")),
        ("float64", "float64"),
        5,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rz.ftz.f32",
        (("mode", "rz"), ("ftz", "ftz"), ("type", "f32")),
        ("float32", "float32"),
        25,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rz.f32",
        (("mode", "rz"), ("ftz", ""), ("type", "f32")),
        ("float32", "float32"),
        26,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rz.f64",
        (("mode", "rz"), ("ftz", ""), ("type", "f64")),
        ("float64", "float64"),
        6,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rm.ftz.f32",
        (("mode", "rm"), ("ftz", "ftz"), ("type", "f32")),
        ("float32", "float32"),
        27,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rm.f32",
        (("mode", "rm"), ("ftz", ""), ("type", "f32")),
        ("float32", "float32"),
        28,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rm.f64",
        (("mode", "rm"), ("ftz", ""), ("type", "f64")),
        ("float64", "float64"),
        7,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rp.ftz.f32",
        (("mode", "rp"), ("ftz", "ftz"), ("type", "f32")),
        ("float32", "float32"),
        29,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rp.f32",
        (("mode", "rp"), ("ftz", ""), ("type", "f32")),
        ("float32", "float32"),
        30,
    ),
    _RuntimeForm(
        SQRT,
        "sqrt.rp.f64",
        (("mode", "rp"), ("ftz", ""), ("type", "f64")),
        ("float64", "float64"),
        8,
    ),
)


def _slug(op_name: str) -> str:
    return op_name.rsplit(".", 1)[-1]


def _make_kernel():
    inputs = sorted(
        {
            (_slug(form.op_name), index, dtype)
            for form in _FORMS
            for index, dtype in enumerate(form.dtypes[1:])
        }
    )
    output_rows: dict[str, int] = {}
    for form in _FORMS:
        output_rows[form.dtypes[0]] = max(output_rows.get(form.dtypes[0], 0), form.output_row + 1)
    parameters = [
        *(f'input_{op}_{index}_{dtype}: T.Buffer((32,), "{dtype}")' for op, index, dtype in inputs),
        *(
            f'output_{dtype}: T.Buffer(({rows}, 32), "{dtype}")'
            for dtype, rows in sorted(output_rows.items())
        ),
    ]
    lines = [
        "@T.prim_func",
        "def ptx_float_register_ops(",
        *(f"    {parameter}," for parameter in parameters),
        "):",
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
    ]
    for form in _FORMS:
        arguments = [f"output_{form.dtypes[0]}[{form.output_row}, lane]"]
        arguments.extend(
            f"input_{_slug(form.op_name)}_{index}_{dtype}[lane]"
            for index, dtype in enumerate(form.dtypes[1:])
        )
        lines.append(f'    T.ptx["{form.spelling}"]({", ".join(arguments)})')
    return tvm.script.from_source("\n".join(lines), {"T": T})


def _resize(values: list[float], dtype: np.dtype) -> np.ndarray:
    return np.resize(np.asarray(values, dtype=dtype), 32)


_RAW_F32 = np.asarray(
    [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x4000_0000,
        0xC000_0000,
        0x4049_0FDB,
        0xC049_0FDB,
        0x7F80_0000,
        0xFF80_0000,
        0x7F81_2345,
        0xFF85_4321,
        0x7FC1_2345,
        0xFFC5_4321,
    ],
    dtype=np.uint32,
)
_RAW_F64 = np.asarray(
    [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
        0x0010_0000_0000_0000,
        0x8010_0000_0000_0000,
        0x3FE0_0000_0000_0000,
        0xBFE0_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xBFF0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0xC000_0000_0000_0000,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x7FF0_1234_5678_9ABC,
        0xFFF0_ABCD_EF01_2345,
        0x7FF8_1234_5678_9ABC,
        0xFFF8_ABCD_EF01_2345,
    ],
    dtype=np.uint64,
)
_HALF_WORDS = np.asarray(
    [
        0x0000,
        0x8000,
        0x0001,
        0x8001,
        0x03FF,
        0x83FF,
        0x0400,
        0x8400,
        0x3C00,
        0xBC00,
        0x7C00,
        0xFC00,
        0x7C01,
        0xFC01,
        0x7E00,
        0xFE00,
        0x7F81,
        0xFF81,
        0x7FC1,
        0xFFC1,
        *((0x9E37 * lane + 0x1234) & 0xFFFF for lane in range(20, 32)),
    ],
    dtype=np.uint16,
)
_PACKED_HALF_WORDS = _HALF_WORDS | (np.roll(_HALF_WORDS, 7).astype(np.uint32) << 16)


def _mad_input(index: int, dtype: str) -> np.ndarray:
    if dtype == "float32":
        tiny = np.float32(np.finfo(np.float32).smallest_subnormal)
        values = (
            [np.nextafter(np.float32(1), np.float32(np.inf)), 1.4, -1.4, tiny, -tiny, 3.25]
            if index == 0
            else (
                [np.nextafter(np.float32(1), np.float32(np.inf)), 1.3, 1.3, 2.0, 2.0, -0.75]
                if index == 1
                else [-1.0, 0.1, -0.1, 0.25, -0.25, 0.625]
            )
        )
        return _resize(values, np.dtype(np.float32))
    values = (
        [math.nextafter(1.0, math.inf), 1.4, -1.4, 3.25]
        if index == 0
        else (
            [math.nextafter(1.0, math.inf), 1.3, 1.3, -0.75]
            if index == 1
            else [-1.0, 0.1, -0.1, 0.625]
        )
    )
    return _resize(values, np.dtype(np.float64))


def _input(op_name: str, index: int, dtype: str) -> np.ndarray:
    if op_name == COPYSIGN:
        words = np.resize(_RAW_F32 if dtype == "float32" else _RAW_F64, 32).copy()
        if index:
            words = np.roll(words, 9)
        return words.view(getattr(np, dtype))
    if op_name == NEG_HALF:
        return (_HALF_WORDS if dtype == "uint16" else _PACKED_HALF_WORDS).copy()
    if op_name == MAD_F:
        return _mad_input(index, dtype)
    words = np.resize(_RAW_F32 if dtype == "float32" else _RAW_F64, 32).copy()
    return words.view(getattr(np, dtype))


@pytest.fixture(scope="module")
def _execution():
    inputs = {
        (form.op_name, index, dtype): _input(form.op_name, index, dtype)
        for form in _FORMS
        for index, dtype in enumerate(form.dtypes[1:])
    }
    output_rows: dict[str, int] = {}
    for form in _FORMS:
        output_rows[form.dtypes[0]] = max(output_rows.get(form.dtypes[0], 0), form.output_row + 1)
    arguments = {
        **{
            f"input_{_slug(op_name)}_{index}_{dtype}": value
            for (op_name, index, dtype), value in inputs.items()
        },
        **{
            f"output_{dtype}": np.zeros((rows, 32), dtype=getattr(np, dtype))
            for dtype, rows in output_rows.items()
        },
    }
    result = v2.Engine().run(
        v2.transpile(_make_kernel()),
        arguments,
        outputs=tuple(f"output_{dtype}" for dtype in output_rows),
    )
    return result, inputs


def _forms_for(op_name: str) -> tuple[_RuntimeForm, ...]:
    return tuple(form for form in _FORMS if form.op_name == op_name)


def _actual(result, form: _RuntimeForm) -> np.ndarray:
    return result.outputs[f"output_{form.dtypes[0]}"][form.output_row]


def _float_bits(values: np.ndarray) -> np.ndarray:
    return values.view(np.uint32 if values.dtype == np.float32 else np.uint64)


def test_copysign_copies_only_the_sign_bit(_execution):
    """Port of ``tests/numsim/runtime/test_ptx_float_register_ops.py::test_copysign_copies_only_the_sign_bit``."""

    result, inputs = _execution
    for form in _forms_for(COPYSIGN):
        sign, magnitude = (
            inputs[form.op_name, index, dtype] for index, dtype in enumerate(form.dtypes[1:])
        )
        sign_bits = _float_bits(sign)
        magnitude_bits = _float_bits(magnitude)
        sign_mask = np.asarray(
            1 << (31 if form.dtypes[0] == "float32" else 63), dtype=magnitude_bits.dtype
        )
        expected = (magnitude_bits & ~sign_mask) | (sign_bits & sign_mask)
        np.testing.assert_array_equal(_float_bits(_actual(result, form)), expected)


def _neg_half_oracle(values: np.ndarray, *, packed: bool, ftz: bool) -> np.ndarray:
    words = values.astype(np.uint32, copy=True)
    shifts = (0, 16) if packed else (0,)
    result = np.zeros_like(words)
    for shift in shifts:
        lane = (words >> shift) & 0xFFFF
        magnitude = lane & 0x7FFF
        if ftz:
            lane = np.where((magnitude != 0) & (magnitude < 0x0400), lane & 0x8000, lane)
        result |= ((lane ^ 0x8000) & 0xFFFF) << shift
    return result.astype(values.dtype)


def test_neg_half_models_finite_values_and_chooses_the_documented_nan_representative(_execution):
    """Port of ``tests/numsim/runtime/test_ptx_float_register_ops.py::test_neg_half_models_finite_values_and_chooses_the_documented_nan_representative``."""

    result, inputs = _execution
    for form in _forms_for(NEG_HALF):
        source = inputs[form.op_name, 0, form.dtypes[1]]
        modifiers = dict(form.modifiers)
        expected = _neg_half_oracle(
            source, packed=modifiers["type"].endswith("x2"), ftz=bool(modifiers["ftz"])
        )
        np.testing.assert_array_equal(_actual(result, form), expected)


def _flush_f32(value: np.float32) -> np.float32:
    bits = int(np.asarray(value).view(np.uint32))
    return (
        np.asarray(bits & 0x8000_0000, dtype=np.uint32).view(np.float32)
        if bits & 0x7FFF_FFFF < 0x0080_0000 and bits & 0x7FFF_FFFF
        else value
    )


def _round_fraction(value: Fraction, dtype: str, mode: str):
    scalar = np.float32 if dtype == "float32" else np.float64
    nearest = scalar(float(value))
    nearest_fraction = Fraction.from_float(float(nearest))
    if nearest_fraction == value:
        return nearest
    if nearest_fraction < value:
        lower = nearest
        upper = np.nextafter(nearest, scalar(np.inf), dtype=scalar)
    else:
        lower = np.nextafter(nearest, scalar(-np.inf), dtype=scalar)
        upper = nearest
    if mode == "rm" or (mode == "rz" and value >= 0):
        return lower
    if mode == "rp" or (mode == "rz" and value < 0):
        return upper
    lower_distance = value - Fraction.from_float(float(lower))
    upper_distance = Fraction.from_float(float(upper)) - value
    if lower_distance < upper_distance:
        return lower
    if upper_distance < lower_distance:
        return upper
    return (
        lower
        if int(np.asarray(lower).view(np.uint32 if dtype == "float32" else np.uint64)) & 1 == 0
        else upper
    )


def _mad_oracle(form: _RuntimeForm, sources: tuple[np.ndarray, ...]) -> np.ndarray:
    modifiers = dict(form.modifiers)
    dtype = form.dtypes[0]
    expected = []
    for lane in range(32):
        values = [source[lane] for source in sources]
        if modifiers["ftz"]:
            values = [_flush_f32(value) for value in values]
        exact = Fraction.from_float(float(values[0])) * Fraction.from_float(
            float(values[1])
        ) + Fraction.from_float(float(values[2]))
        rounded = _round_fraction(exact, dtype, modifiers["rnd"])
        if modifiers["ftz"]:
            rounded = _flush_f32(rounded)
        if modifiers["sat"]:
            rounded = type(rounded)(min(1.0, max(0.0, float(rounded))))
        expected.append(rounded)
    return np.asarray(expected, dtype=getattr(np, dtype))


def test_mad_f_reuses_exact_fused_arithmetic_for_every_supported_modifier(_execution):
    """Port of ``tests/numsim/runtime/test_ptx_float_register_ops.py::test_mad_f_reuses_exact_fused_arithmetic_for_every_supported_modifier``."""

    result, inputs = _execution
    for form in _forms_for(MAD_F):
        sources = tuple(
            inputs[form.op_name, index, dtype] for index, dtype in enumerate(form.dtypes[1:])
        )
        np.testing.assert_array_equal(
            _float_bits(_actual(result, form)), _float_bits(_mad_oracle(form, sources))
        )


def _sqrt_oracle(value, dtype: str, mode: str, ftz: bool):
    scalar = np.float32 if dtype == "float32" else np.float64
    if ftz:
        value = _flush_f32(value)
    if np.isnan(value) or (value < 0.0 and not (value == 0.0)):
        return scalar(np.nan)
    nearest = scalar(math.sqrt(float(value)))
    if mode not in {"rz", "rm", "rp"} or not np.isfinite(value) or value == 0.0:
        return _flush_f32(nearest) if ftz else nearest
    square = Fraction.from_float(float(nearest)) ** 2
    exact = Fraction.from_float(float(value))
    if mode in {"rz", "rm"} and square > exact:
        nearest = np.nextafter(nearest, scalar(-np.inf), dtype=scalar)
    elif mode == "rp" and square < exact:
        nearest = np.nextafter(nearest, scalar(np.inf), dtype=scalar)
    return _flush_f32(nearest) if ftz else nearest


def test_sqrt_all_rounding_modes_and_ftz_forms_match_an_exact_square_oracle(_execution):
    """Port of ``tests/numsim/runtime/test_ptx_float_register_ops.py::test_sqrt_all_rounding_modes_and_ftz_forms_match_an_exact_square_oracle``."""

    result, inputs = _execution
    for form in _forms_for(SQRT):
        source = inputs[form.op_name, 0, form.dtypes[1]]
        modifiers = dict(form.modifiers)
        expected = np.asarray(
            [
                _sqrt_oracle(value, form.dtypes[0], modifiers["mode"], bool(modifiers["ftz"]))
                for value in source
            ],
            dtype=getattr(np, form.dtypes[0]),
        )
        actual = _actual(result, form)
        assert np.array_equal(np.isnan(actual), np.isnan(expected))
        finite = ~np.isnan(expected)
        np.testing.assert_array_equal(_float_bits(actual[finite]), _float_bits(expected[finite]))


@pytest.mark.parametrize("op_name", (SIN, COS), ids=("sin", "cos"))
def test_trigonometric_approximations_obey_special_values_ftz_and_error_bound(
    _execution, op_name: str
):
    """Port of ``tests/numsim/runtime/test_ptx_float_register_ops.py::test_trigonometric_approximations_obey_special_values_ftz_and_error_bound`` (both ``sin``/``cos`` parametrizations)."""

    result, inputs = _execution
    operation = math.sin if op_name == SIN else math.cos
    for form in _forms_for(op_name):
        source = inputs[form.op_name, 0, form.dtypes[1]]
        ftz = bool(dict(form.modifiers)["ftz"])
        expected = []
        for value in source:
            value = _flush_f32(value) if ftz else value
            computed = np.float32(np.nan if not np.isfinite(value) else operation(float(value)))
            expected.append(_flush_f32(computed) if ftz else computed)
        expected = np.asarray(expected, dtype=np.float32)
        actual = _actual(result, form)
        assert np.array_equal(np.isnan(actual), np.isnan(expected))
        finite = ~np.isnan(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.0, atol=2**-20.5)
        zero = finite & (expected == 0.0)
        np.testing.assert_array_equal(_float_bits(actual[zero]), _float_bits(expected[zero]))
