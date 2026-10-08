"""Host-side TensorMap image decoding and host-array layout for test fixtures.

Test fixtures (corpus cases, the GPU microtest harness, three-way runs) need
to read back the ``numsim.cases.TensorMap`` descriptor images they build and to
lay out aliased host arrays for a GPU launch. These are plain-NumPy ports of
what the fixtures used from the legacy ``numsim.bindings`` module, kept here so
the fixtures do not depend on the legacy execution path. They only read the
kept ``numsim.cases`` descriptor format; they never bind anything for NumSim.
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from tirx_harness.numsim.cases import (
    _TENSOR_MAP_DESCRIPTOR_BYTES,
    _TENSOR_MAP_DTYPE_CODES,
    _TENSOR_MAP_FLAG_MAGIC,
    _TENSOR_MAP_FORMAT_TAGS,
    _TENSOR_MAP_SWIZZLES,
    Im2col,
    _buffer_dtype_itemsize,
    _tensor_map_element_bits,
    _tensor_map_payload_bytes,
)

PACKED_FLOAT4_DTYPE = "float4_e2m1fn"
_TENSOR_MAP_DTYPES = tuple(sorted(_TENSOR_MAP_DTYPE_CODES, key=_TENSOR_MAP_DTYPE_CODES.__getitem__))


@dataclass(frozen=True)
class DecodedTensorMap:
    """One TensorMap image found in a host array (descriptor fields)."""

    byte_offset: int
    address: int
    required_byte_len: int
    global_shape: tuple[int, ...]
    global_strides: tuple[int, ...]
    box_shape: tuple[int, ...]
    element_strides: tuple[int, ...]
    dtype: str
    fp4_shared_layout: str | None
    swizzle: str | None
    inactive_swizzle_atomicity: int
    fill_mode: str | None
    interleave_bytes: int | None
    im2col: Im2col | None

    @property
    def physical_global_shape(self) -> tuple[int, ...]:
        """Interleaved dimension zero counts slices, not scalar elements."""
        if self.interleave_bytes is None:
            return self.global_shape
        element_bits = _tensor_map_element_bits(self.dtype)
        elements = self.interleave_bytes * 8 // element_bits
        return (self.global_shape[0] * elements, *self.global_shape[1:])


def decode_tensor_maps(array: np.ndarray) -> tuple[DecodedTensorMap, ...]:
    """Every valid TensorMap image at a 128-byte boundary of ``array``."""

    if not array.flags.c_contiguous:
        return ()
    flat = array.view(np.uint8).reshape(-1)
    result: list[DecodedTensorMap] = []
    for offset in range(0, flat.size - _TENSOR_MAP_DESCRIPTOR_BYTES + 1, 128):
        image = flat[offset : offset + _TENSOR_MAP_DESCRIPTOR_BYTES]
        tag = int(image[63])
        if tag & ~0x58 not in _TENSOR_MAP_FORMAT_TAGS.values():
            continue
        payload_bytes = _tensor_map_payload_bytes(tag)
        if np.any(image[payload_bytes:]):
            continue
        flags = int(image[60])
        if flags & 0x80 != _TENSOR_MAP_FLAG_MAGIC or flags & (1 << 5) == 0:
            continue
        if int.from_bytes(image[8:16].tobytes(), "little") != 0:
            continue
        rank = int(image[59]) & 0b111
        dtype_code = int(image[59]) >> 3
        if rank == 0 or rank > 5 or dtype_code >= len(_TENSOR_MAP_DTYPES):
            continue
        global_shape: list[int] = []
        for axis in range(rank):
            start = 16 + axis * 4
            encoded = int.from_bytes(image[start : start + 4].tobytes(), "little")
            global_shape.append(2**32 if encoded == 0 else encoded)
        if any(
            int.from_bytes(image[16 + axis * 4 : 20 + axis * 4].tobytes(), "little") != 1
            for axis in range(rank, 5)
        ):
            continue
        global_strides: list[int] = []
        for pair in range(2):
            start = 36 + pair * 9
            packed = int.from_bytes(image[start : start + 9].tobytes(), "little")
            global_strides.extend(((packed & ((1 << 36) - 1)) << 4, (packed >> 36) << 4))
        if any(global_strides[rank - 1 :]):
            continue
        dtype = _TENSOR_MAP_DTYPES[dtype_code]
        element_bits = _tensor_map_element_bits(dtype)
        interleave_bytes = next(
            width for width, base_tag in _TENSOR_MAP_FORMAT_TAGS.items() if base_tag == tag & ~0x58
        )
        if interleave_bytes is not None and rank < 3:
            continue
        transfer_bits = interleave_bytes * 8 if interleave_bytes is not None else element_bits
        required_byte_len = (global_shape[0] * transfer_bits + 7) // 8
        for stride, dimension in zip(global_strides[: rank - 1], global_shape[1:], strict=True):
            required_byte_len += (dimension - 1) * stride
        address = int.from_bytes(image[0:8].tobytes(), "little")
        if address == 0:
            continue
        fp4_shared_layout = {0: None, 1: "align8_packed", 2: "align16_padded"}.get(flags & 0b11)
        if fp4_shared_layout is None and flags & 0b11:
            continue
        if (dtype == PACKED_FLOAT4_DTYPE) != (fp4_shared_layout is not None):
            continue
        swizzle_code = ((flags >> 2) & 0b11) | ((flags >> 4) & 4)
        atomicity = ((tag >> 3) & 1) | ((tag >> 5) & 2)
        swizzle = None
        if swizzle_code:
            try:
                swizzle = next(
                    name for name, code in _TENSOR_MAP_SWIZZLES.items()
                    if code == (swizzle_code, atomicity)
                )
            except StopIteration:
                continue
        box_shape = tuple(int(image[54 + axis]) + 1 for axis in range(rank))
        im2col = None
        if tag & 0x10:
            if rank < 3 or np.any(image[56:59]) or np.any(image[77:80]) or int(image[76]) & ~7:
                continue
            box_shape = (box_shape[0], box_shape[1] + ((int(image[76]) & 3) << 8))
            wide = bool(int(image[76]) & 4)
            spatial_rank = 1 if wide else rank - 2
            corners = struct.unpack_from("<3h3h", image, 64)
            im2col = Im2col(corners[:spatial_rank], corners[3 : 3 + spatial_rank], wide)
        elif np.any(image[54 + rank : 59]):
            continue
        packed_element_strides = int.from_bytes(image[61:63].tobytes(), "little")
        if packed_element_strides >> (rank * 3):
            continue
        element_strides = tuple(
            ((packed_element_strides >> (axis * 3)) & 0b111) + 1 for axis in range(rank)
        )
        result.append(
            DecodedTensorMap(
                byte_offset=offset,
                address=address,
                required_byte_len=required_byte_len,
                global_shape=tuple(global_shape),
                global_strides=tuple(global_strides[: rank - 1]),
                box_shape=box_shape,
                element_strides=element_strides,
                dtype=dtype,
                fp4_shared_layout=fp4_shared_layout,
                swizzle=swizzle,
                inactive_swizzle_atomicity=atomicity if not swizzle_code else 0,
                fill_mode="nan" if flags & (1 << 4) else None,
                interleave_bytes=interleave_bytes,
                im2col=im2col,
            )
        )
    return tuple(result)


def tensor_map_physical_dtype(dtype: str) -> np.dtype[Any]:
    """The NumPy storage dtype of a TensorMap element dtype."""

    return np.dtype(
        {
            PACKED_FLOAT4_DTYPE: "uint8",
            "uint6": "uint8",
            "float8_e4m3fn": "uint8",
            "float8_e8m0fnu": "uint8",
            "bfloat16": "uint16",
            "tf32": "float32",
            "float32_ftz": "float32",
            "tf32_ftz": "float32",
            "uint32x2": "uint64",
        }.get(dtype, dtype)
    )


def tensor_map_array_from_buffer(
    buffer: Any,
    *,
    data_offset: int,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    """The physical NumPy view described by TensorMap fields over ``buffer``."""

    physical_dtype = tensor_map_physical_dtype(dtype)
    bits = _tensor_map_element_bits(dtype)
    dimension_zero = (global_shape[0] * bits + 7) // 8 if bits < 8 else global_shape[0]
    coordinate_shape = (dimension_zero, *global_shape[1:])
    coordinate_strides = (int(physical_dtype.itemsize), *global_strides)
    return np.ndarray(
        shape=tuple(reversed(coordinate_shape)),
        dtype=physical_dtype,
        buffer=buffer,
        offset=data_offset,
        strides=tuple(reversed(coordinate_strides)),
    )


def tensor_map_base_array(descriptor: np.ndarray) -> np.ndarray:
    """The host tensor (a view, descriptor dimension order) one image addresses."""

    decoded = decode_tensor_maps(np.asarray(descriptor))
    if np.asarray(descriptor).shape != (128,) or len(decoded) != 1:
        raise ValueError("expected one uint8[128] TensorMap descriptor")
    tensor_map = decoded[0]
    raw = np.ctypeslib.as_array(
        (ctypes.c_uint8 * tensor_map.required_byte_len).from_address(tensor_map.address)
    )
    return tensor_map_array_from_buffer(
        raw,
        data_offset=0,
        global_shape=tensor_map.physical_global_shape,
        global_strides=tensor_map.global_strides,
        dtype=tensor_map.dtype,
    )


# --- host-array layout (alias groups) for a GPU launch -----------------------


@dataclass(frozen=True)
class HostAllocation:
    """One physical host byte range shared by connected arrays."""

    data: bytes
    host_address: int

    @property
    def byte_len(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class HostBufferView:
    """One named array inside a :class:`HostAllocation`."""

    allocation: int
    data_offset: int
    dtype: str
    itemsize: int
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]


@dataclass(frozen=True)
class HostLayout:
    allocations: tuple[HostAllocation, ...]
    buffers: dict[str, HostBufferView]


@dataclass(frozen=True)
class _Pending:
    name: str
    array: np.ndarray
    dtype: str
    itemsize: int
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    origin: int
    low: int
    high: int
    owner: Any
    backing_low: int
    backing_high: int
    public: bool


def _byte_bounds(
    origin: int, shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> tuple[int, int]:
    if any(extent < 0 for extent in shape):
        raise ValueError(f"buffer shape contains a negative extent: {shape}")
    if any(extent == 0 for extent in shape):
        return origin, origin
    low = origin
    high = origin
    for extent, stride in zip(shape, strides):
        delta = (extent - 1) * stride
        if delta < 0:
            low += delta
        else:
            high += delta
    return low, high + itemsize


def _array_owner(array: np.ndarray) -> tuple[Any, np.ndarray]:
    owner: Any = array
    root_array = array
    seen: set[int] = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        if isinstance(owner, np.ndarray):
            root_array = owner
        base = getattr(owner, "base", None)
        if base is None and isinstance(owner, memoryview):
            base = owner.obj
        if base is None:
            break
        owner = base
    return owner, root_array


def _array_pointer(array: np.ndarray) -> int:
    pointer = int(array.__array_interface__["data"][0])
    if pointer == 0 and array.size:
        raise ValueError("NumPy array exposes a null data pointer")
    return pointer


def _pending(name: str, array: np.ndarray, *, expected_dtype: str | None, public: bool) -> _Pending:
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise ValueError(f"host buffer {name!r} cannot use object dtype")
    dtype = str(array.dtype) if expected_dtype is None else expected_dtype
    if dtype == PACKED_FLOAT4_DTYPE:
        if array.dtype != np.dtype(np.uint8) or not array.flags.c_contiguous:
            raise ValueError(f"packed float4 buffer {name!r} requires a contiguous uint8 array")
        itemsize = 1
    else:
        itemsize = _buffer_dtype_itemsize(dtype)
        if itemsize is None:
            raise ValueError(f"unsupported host buffer dtype {dtype!r}")
        if itemsize != int(array.dtype.itemsize):
            raise ValueError(
                f"declared dtype {dtype!r} has itemsize {itemsize}, but host array "
                f"dtype {array.dtype} has itemsize {array.dtype.itemsize}"
            )
    packed_float4 = dtype == PACKED_FLOAT4_DTYPE
    shape = (array.size * 2,) if packed_float4 else tuple(int(v) for v in array.shape)
    byte_strides = () if packed_float4 else tuple(int(v) for v in array.strides)
    owner, root_array = _array_owner(array)
    origin = _array_pointer(array)
    root_low, root_high = _byte_bounds(
        _array_pointer(root_array),
        tuple(int(v) for v in root_array.shape),
        tuple(int(v) for v in root_array.strides),
        int(root_array.dtype.itemsize),
    )
    low, high = (
        (origin, origin + array.nbytes)
        if packed_float4
        else _byte_bounds(origin, shape, byte_strides, itemsize)
    )
    if low < root_low or high > root_high:
        raise ValueError(
            f"host buffer {name!r} byte range [{low}, {high}) escapes its "
            f"NumPy backing range [{root_low}, {root_high})"
        )
    return _Pending(
        name=name, array=array, dtype=dtype, itemsize=itemsize, shape=shape,
        byte_strides=byte_strides, origin=origin, low=low, high=high, owner=owner,
        backing_low=root_low, backing_high=root_high, public=public,
    )


def _tensor_map_owner_array(descriptor_array: np.ndarray, descriptor: DecodedTensorMap) -> np.ndarray | None:
    base = getattr(descriptor_array, "_tensor_map_base", None)
    if not isinstance(base, np.ndarray):
        return None
    _owner, root_array = _array_owner(base)
    root_low, root_high = _byte_bounds(
        _array_pointer(root_array),
        tuple(int(v) for v in root_array.shape),
        tuple(int(v) for v in root_array.strides),
        int(root_array.dtype.itemsize),
    )
    if not (root_low <= descriptor.address and descriptor.address + descriptor.required_byte_len <= root_high):
        return None
    return root_array


def host_layout(
    arrays: dict[str, np.ndarray], *, expected_dtypes: dict[str, str] | None = None
) -> HostLayout:
    """Group host arrays into physical allocations by NumPy connectivity.

    Arrays sharing a NumPy owner or overlapping backing ranges, plus the
    memory any embedded TensorMap image addresses, form one allocation; each
    allocation holds a snapshot of its host bytes. This is the layout the
    legacy binder derived (``prepare_bindings(...).allocations/.buffers``),
    without binding anything for NumSim.
    """

    expected_dtypes = expected_dtypes or {}
    pending: list[_Pending] = []
    decoded_by_name: dict[str, tuple[DecodedTensorMap, ...]] = {}
    for name in sorted(arrays):
        value = arrays[name]
        pending.append(_pending(name, value, expected_dtype=expected_dtypes.get(name), public=True))
        decoded_by_name[name] = decode_tensor_maps(np.asarray(value))
    hidden = 0
    for name, descriptors in decoded_by_name.items():
        for descriptor in descriptors:
            carrier = _tensor_map_owner_array(arrays[name], descriptor)
            if carrier is None:
                carrier = np.ctypeslib.as_array(
                    (ctypes.c_uint8 * descriptor.required_byte_len).from_address(descriptor.address)
                )
            pending.append(_pending(f"__tensor_map_base__:{hidden}", carrier, expected_dtype=None, public=False))
            hidden += 1

    parent = list(range(len(pending)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(pending):
        for right_index in range(left_index + 1, len(pending)):
            right = pending[right_index]
            if left.owner is right.owner or (
                left.backing_low < right.backing_high and right.backing_low < left.backing_high
            ):
                left_root, right_root = find(left_index), find(right_index)
                if left_root != right_root:
                    parent[right_root] = left_root
    groups: dict[int, list[_Pending]] = {}
    for index, item in enumerate(pending):
        groups.setdefault(find(index), []).append(item)

    allocations: list[HostAllocation] = []
    buffers: dict[str, HostBufferView] = {}
    for items in groups.values():
        low = min(item.low for item in items)
        high = max(item.high for item in items)
        data = (
            bytes((ctypes.c_uint8 * (high - low)).from_address(low)) if high > low else b""
        )
        index = len(allocations)
        allocations.append(HostAllocation(data=data, host_address=low))
        for item in items:
            if item.public:
                buffers[item.name] = HostBufferView(
                    allocation=index, data_offset=item.origin - low, dtype=item.dtype,
                    itemsize=item.itemsize, shape=item.shape, byte_strides=item.byte_strides,
                )
    return HostLayout(allocations=tuple(allocations), buffers=buffers)


__all__ = [
    "PACKED_FLOAT4_DTYPE",
    "DecodedTensorMap",
    "HostAllocation",
    "HostBufferView",
    "HostLayout",
    "decode_tensor_maps",
    "host_layout",
    "tensor_map_array_from_buffer",
    "tensor_map_base_array",
    "tensor_map_physical_dtype",
]
