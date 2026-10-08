"""Dtype helpers: TVM dtype strings, bit widths and constant encoding."""

from __future__ import annotations

import re
import struct
from functools import cache
from typing import Any

_SCALAR_BITS = {
    "bool": 8, "int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32,
    "uint32": 32, "int64": 64, "uint64": 64, "uint128": 128, "int128": 128,
    "float16": 16, "bfloat16": 16, "float32": 32, "float64": 64,
    "float8_e4m3fn": 8, "float8_e5m2": 8, "float8_e8m0fnu": 8, "float8_e3m4": 8,
    "float8_e4m3": 8, "float8_e4m3fnuz": 8, "float8_e5m2fnuz": 8,
    "float6_e2m3fn": 6, "float6_e3m2fn": 6, "float4_e2m1fn": 4,
    "uint6": 6, "tf32": 32, "handle": 64, "float8_e4m3b11fnuz": 8,
}

_VECTOR = re.compile(r"^(.*?)x(\d+)$")


def type_key(node: Any) -> str:
    info = getattr(type(node), "__tvm_ffi_type_info__", None)
    return info.type_key if info is not None else type(node).__name__


@cache
def split(dtype: str) -> tuple[str, int]:
    """``"float16x2" -> ("float16", 2)``; scalars have one lane."""
    if dtype in _SCALAR_BITS:
        return dtype, 1
    match = _VECTOR.match(dtype)
    if match and match.group(1) in _SCALAR_BITS:
        return match.group(1), int(match.group(2))
    return dtype, 1


def bits(dtype: str) -> int:
    element, lanes = split(dtype)
    if element not in _SCALAR_BITS:
        raise ValueError(f"unknown dtype {dtype!r}")
    return _SCALAR_BITS[element] * lanes


def known(dtype: str) -> bool:
    return split(dtype)[0] in _SCALAR_BITS


def is_float(dtype: str) -> bool:
    element = split(dtype)[0]
    return element.startswith(("float", "bfloat", "tf32"))


def is_signed(dtype: str) -> bool:
    return split(dtype)[0].startswith("int")


def dtype_of(node: Any) -> str:
    """The dtype of an expression or variable; ``handle`` for pointers, ``""`` for void."""
    ty = getattr(node, "ty", None)
    if ty is None:
        return ""
    key = type_key(ty)
    if key == "ir.PointerType":
        return "handle"
    if key == "tirx.BufferType":
        return "handle"
    if key == "tirx.TensorMapType":
        return "handle"
    dtype = getattr(ty, "dtype", None)
    if dtype is None:
        return ""
    text = str(dtype)
    return "" if text == "void" else text


def _float_bits(element: str, value: float) -> int:
    if element == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if element == "float64":
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    if element == "float16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    import ml_dtypes
    import numpy as np

    scalar = getattr(ml_dtypes, element, None)
    if scalar is None:
        raise ValueError(f"constant of dtype {element} is not encodable")
    raw = np.array([value], dtype=scalar).view(np.uint8 if np.dtype(scalar).itemsize == 1 else np.uint16)
    return int(raw[0]) & ((1 << _SCALAR_BITS[element]) - 1)


def encode(dtype: str, value: int | float) -> int:
    """Bit pattern of ``value`` in ``dtype``, broadcast over vector lanes."""
    element, lanes = split(dtype)
    if element not in _SCALAR_BITS:
        raise ValueError(f"constant of dtype {dtype} is not encodable")
    width = _SCALAR_BITS[element]
    if is_float(element):
        lane = _float_bits(element, float(value))
    elif element == "bool":
        lane = int(bool(value))
    else:
        lane = int(value) & ((1 << width) - 1)
    result = 0
    for index in range(lanes):
        result |= lane << (index * width)
    return result


__all__ = ["bits", "dtype_of", "encode", "is_float", "is_signed", "known", "split", "type_key"]
