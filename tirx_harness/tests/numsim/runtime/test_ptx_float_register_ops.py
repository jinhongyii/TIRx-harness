from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest
import tvm
from tvm.backend.cuda.ptx.table import canonical_dtypes, mods, variants
from tvm.script import tirx as T

from tirx_harness import numsim

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


def _slug(op_name: str) -> str:
    return op_name.rsplit(".", 1)[-1]


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


def _actual(result, form: _RuntimeForm) -> np.ndarray:
    return result.outputs[f"output_{form.dtypes[0]}"][form.output_row]


def _float_bits(values: np.ndarray) -> np.ndarray:
    return values.view(np.uint32 if values.dtype == np.float32 else np.uint64)


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


