"""Engine: inputs -> ``numsim_core_py.run`` -> results (plan 2.7 ``run.py``).

``Engine`` keeps the legacy signature (``Engine(max_workers=...)``,
``.run``, ``.run_racecheck_phase``, ``.run_synccheck_phase``). Inputs are
canonicalized and validated in exactly one function,
:func:`canonicalize_inputs`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import re

import numpy as np

from .compile import CompiledModule, native
from .options import BACKENDS, options
from .report import (
    AnalysisResult,
    attach_operations,
    checker_phase_payload,
    NumSimResult,
    diagnostic_from_core,
    phase_payload,
    record_from_core,
)


from tirx_harness.numsim.errors import NumSimExecutionError


class InputError(NumSimExecutionError, ValueError):
    """Inputs do not match the module's host ABI."""


class MissingBindingsError(InputError):
    """Required host bindings are absent (``missing`` lists them)."""

    def __init__(self, missing: list[str]):
        super().__init__(f"NumSim inputs are missing required bindings: {missing}")
        self.missing = list(missing)


class ExecutionError(NumSimExecutionError):
    """A NumSim run did not complete (runtime error, deadlock or fail-closed
    incomplete); ``diagnostics`` holds the engine's records."""

    def __init__(self, message: str, diagnostics: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _raise_unless_completed(status: Mapping[str, Any], diagnostics: list[dict[str, Any]]) -> None:
    """Plain NumSim cannot return outputs of a run that stopped early
    (legacy raised ``NumSimExecutionError`` too)."""

    if status.get("kind") in (None, "completed"):
        return
    stops = [d for d in diagnostics if d.get("status") in ("error", "incomplete")
             and d.get("reason") != "subset_execution"]
    if not stops:
        return
    first = stops[0]
    where = ""
    span = first.get("source_span")
    if isinstance(span, Mapping):
        leaf = span["spans"][-1] if span.get("kind") == "sequential" else span
        where = f" at {leaf.get('source_name')}:{leaf.get('line')}"
    detail = first.get("message") or first.get("reason") or ""
    raise ExecutionError(
        f"NumSim execution {status.get('kind')}: {first.get('kind')}: {detail}{where}", diagnostics
    )


@dataclass(frozen=True)
class BoundInput:
    """One canonical argument: the native form plus the host layout."""

    name: str
    kind: str  # buffer | scalar | tensor_map | pointer
    native: tuple
    dtype: np.dtype | None = None
    shape: tuple[int, ...] | None = None
    # Tensor maps over a host array, and views of aliased host memory: the
    # buffer argument holding the bytes.
    base: str | None = None
    # Host data pointer of a buffer (synthetic engine addresses keep its low
    # 8 bits, `arena::addr` ruling).
    host_addr: int | None = None
    # A descriptor image bound to a plain buffer parameter: the pointer at
    # bytes [0, 8) is rewritten to (engine address of `base`) + offset.
    patch_offset: int | None = None


_SCALAR_FORMATS = {
    "F32": "<f", "F64": "<d", "TF32": "<f",
    "U8": "<B", "U16": "<H", "U32": "<I", "U64": "<Q",
    "S8": "<b", "S16": "<h", "S32": "<i", "S64": "<q",
}


def _slot_kind(slot: Mapping[str, Any]) -> str:
    kind = slot.get("kind", "Buffer")
    if isinstance(kind, Mapping):  # ParamKind::ImplicitShape { buffer, axis }
        kind = next(iter(kind))
    return re.sub(r"(?<!^)(?=[A-Z])", "_", str(kind)).lower()


def _scalar_bits(name: str, value: Any, ty: Mapping[str, Any] | None) -> int:
    elem = str((ty or {}).get("elem", "S64")).upper()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or elem == "PRED":
        return int(bool(value))
    if elem == "F16":
        return int(np.array(value, dtype=np.float16).view(np.uint16))
    if elem == "BF16":
        return int(np.array(value, dtype=np.float32).view(np.uint32)) >> 16
    fmt = _SCALAR_FORMATS.get(elem)
    if fmt is None:
        raise InputError(f"scalar argument {name!r} has unsupported dtype {elem}")
    try:
        packed = struct.pack(fmt, value)
    except struct.error as error:
        raise InputError(f"scalar argument {name!r}={value!r} does not fit {elem}: {error}") from error
    return int.from_bytes(packed, "little")


# ml_dtypes types that store one sub-byte value per byte.
_UNPACKED_SUB_BYTE = frozenset({"float4_e2m1fn", "int4", "uint4", "int2", "uint2",
                                "float6_e2m3fn", "float6_e3m2fn"})
_SUB_BYTE_ELEMS = frozenset({"E2M1", "U4", "S4", "E2M3", "E3M2", "S2F6"})


def _reject_unpacked_sub_byte(name: str, array: np.ndarray, slot: Mapping[str, Any]) -> None:
    """Sub-byte buffers must be bound packed (legacy: a contiguous uint8
    array); a one-value-per-byte ml_dtypes array would silently be read as
    packed bytes."""

    elem = str((slot.get("dtype") or {}).get("elem", "")).upper()
    if array.dtype.name in _UNPACKED_SUB_BYTE or (elem in _SUB_BYTE_ELEMS and array.dtype.itemsize != 1):
        raise InputError(
            f"buffer argument {name!r} holds sub-byte {elem or array.dtype.name} values one per byte "
            f"({array.dtype}); bind the packed bytes as a contiguous uint8 array"
        )


def canonicalize_inputs(module: CompiledModule, inputs: Mapping[str, Any]) -> dict[str, BoundInput]:
    """Bind user inputs to the module's host ABI (all kernels, by name).

    Accepts canonical names, local names and aliases. Buffers are any
    array-like (made C-contiguous), scalars Python/numpy numbers encoded per
    the slot dtype, tensor maps a 128-byte image, pointers
    ``("pointer", target, offset)``. Unknown names, duplicates and missing
    non-implicit bindings raise :class:`InputError`.
    """

    slots: dict[str, Mapping[str, Any]] = {}
    lookup: dict[str, str] = {}
    for kernel in module.spec.kernels:
        for slot in kernel.host_abi:
            canonical = str(slot["name"])
            slots.setdefault(canonical, slot)
            for alias in (canonical, slot.get("local_name"), *(slot.get("aliases") or ())):
                if alias:
                    # `k<i>:<name>` selects kernel i's binding (legacy form).
                    lookup.setdefault(f"k{kernel.index}:{alias}", canonical)
                    previous = lookup.setdefault(str(alias), canonical)
                    if previous != canonical:
                        lookup[str(alias)] = ""  # ambiguous across kernels
    bound: dict[str, BoundInput] = {}
    tensor_map_bases: dict[int, str] = {}
    given_objects: dict[str, Any] = {}

    def register_base(owner: str, base: np.ndarray) -> str:
        """The buffer argument holding a host descriptor's base array."""

        base_name = tensor_map_bases.get(id(base))
        if base_name is None:
            base_name = next(
                (n for n, given in inputs.items() if given is base and lookup.get(n)),
                f"{owner}.__base__",
            )
            base_name = lookup.get(base_name, base_name)
            tensor_map_bases[id(base)] = base_name
            if base_name not in bound:
                given_objects.setdefault(base_name, base)
                contiguous = np.ascontiguousarray(base)
                bound[base_name] = BoundInput(
                    base_name, "buffer", ("buffer", contiguous.view(np.uint8).reshape(-1).tobytes(), None),
                    dtype=contiguous.dtype, shape=tuple(contiguous.shape),
                    host_addr=int(base.__array_interface__["data"][0]),
                )
        return base_name

    for given, value in inputs.items():
        canonical = lookup.get(given)
        if canonical is None:
            raise InputError(f"NumSim input binding {given!r} is unknown; expected one of {sorted(slots)}")
        if canonical == "":
            raise InputError(f"NumSim input binding {given!r} is ambiguous across kernels")
        if canonical in given_objects:
            # Kernel-qualified names (`k0:x`, `k1:x`) of one shared binding
            # may repeat when they carry the same value.
            if given_objects[canonical] is value:
                continue
            previous = bound[canonical]
            if not (isinstance(value, np.ndarray) and previous.kind == "buffer"
                    and np.ascontiguousarray(value).view(np.uint8).tobytes() == previous.native[1]):
                raise InputError(f"NumSim input {canonical!r} is bound more than once with different values")
            continue
        given_objects[canonical] = value
        slot = slots[canonical]
        kind = _slot_kind(slot)
        if isinstance(value, tuple) and value and value[0] == "pointer":
            bound[canonical] = BoundInput(canonical, "pointer", ("pointer", str(value[1]), int(value[2])))
        elif kind in ("scalar", "implicit_shape"):
            bound[canonical] = BoundInput(canonical, "scalar", ("scalar", _scalar_bits(canonical, value, slot.get("dtype"))))
        elif kind == "tensor_map":
            image = np.ascontiguousarray(np.asarray(value)).view(np.uint8).reshape(-1)
            if image.size != 128:
                raise InputError(f"tensor map {canonical!r} must be a 128-byte image, got {image.size} bytes")
            base = getattr(value, "_tensor_map_base", None)
            if base is None:
                bound[canonical] = BoundInput(canonical, "tensor_map", ("tensor_map", image.tobytes()))
                continue
            # A host descriptor (numsim.cases.TensorMap) addresses a host
            # array: bind that array as a buffer and let the engine encode the
            # map against its address (W8-3 ``TensorMapOf``).
            base = np.asarray(base)
            base_name = register_base(canonical, base)
            pointer = int.from_bytes(image[0:8].tobytes(), "little")
            offset = pointer - int(base.__array_interface__["data"][0])
            if not 0 <= offset <= base.nbytes:
                raise InputError(f"tensor map {canonical!r} does not address its base array")
            bound[canonical] = BoundInput(
                canonical, "tensor_map", ("tensor_map_of", base_name, offset, image.tobytes()), base=base_name,
            )
        else:
            if hasattr(value, "detach") and hasattr(value, "cpu"):  # torch tensor
                value = value.detach().cpu().numpy()
            host = np.asarray(value)
            array = np.ascontiguousarray(host)
            if array.dtype == object:
                raise InputError(f"buffer argument {canonical!r} is not a numeric array")
            _reject_unpacked_sub_byte(canonical, array, slot)
            descriptor_base = getattr(value, "_tensor_map_base", None)
            patch = base_name = None
            if descriptor_base is not None and array.nbytes == 128:
                # A host descriptor image passed to a plain uint8[128]
                # parameter: its pointer must name the engine address of the
                # base array, not the host pointer.
                descriptor_base = np.asarray(descriptor_base)
                base_name = register_base(canonical, descriptor_base)
                pointer = int.from_bytes(array.view(np.uint8)[0:8].tobytes(), "little")
                patch = pointer - int(descriptor_base.__array_interface__["data"][0])
                if not 0 <= patch <= descriptor_base.nbytes:
                    raise InputError(f"descriptor {canonical!r} does not address its base array")
            bound[canonical] = BoundInput(
                canonical, "buffer", ("buffer", array.view(np.uint8).reshape(-1).tobytes(), None),
                dtype=array.dtype, shape=tuple(array.shape),
                host_addr=int(host.__array_interface__["data"][0]) if host.size else None,
                base=base_name, patch_offset=patch,
            )
    # Buffer slot shapes: check static dims, bind `Param` dims (W1 binder
    # convention: implicit shape variables are Scalar slots referenced from
    # the buffer slot's `shape`).
    for kernel in module.spec.kernels:
        abi = kernel.host_abi
        for slot in abi:
            source = bound.get(str(slot["name"]))
            if source is None or source.kind != "buffer" or not slot.get("shape"):
                continue
            _bind_shape(slot, abi, source, bound)
    # ImplicitShape slots bind from the referenced buffer's array shape.
    for kernel in module.spec.kernels:
        abi = kernel.host_abi
        for slot in abi:
            kind = slot.get("kind")
            if not (isinstance(kind, Mapping) and "ImplicitShape" in kind):
                continue
            name = str(slot["name"])
            buffer = str(abi[int(kind["ImplicitShape"]["buffer"])]["name"])
            axis = int(kind["ImplicitShape"]["axis"])
            source = bound.get(buffer)
            if source is None or source.shape is None:
                continue  # reported as missing below
            if axis >= len(source.shape):
                raise InputError(f"{name!r}: buffer {buffer!r} has rank {len(source.shape)}, needs axis {axis}")
            value = _scalar_bits(name, int(source.shape[axis]), slot.get("dtype"))
            if name in bound and bound[name].native[1] != value:
                raise InputError(f"{name!r} disagrees with {buffer!r}.shape[{axis}]={source.shape[axis]}")
            bound[name] = BoundInput(name, "scalar", ("scalar", value))
    _resolve_aliased_buffers(bound, given_objects)
    _patch_descriptor_pointers(module, bound)
    # A tensor-map parameter with no host value and no engine-encodable spec
    # (legacy accepted kernels that never use it): bind an all-zero image,
    # which the descriptor decoder rejects, so any use fails closed.
    for name, slot in slots.items():
        if name not in bound and _slot_kind(slot) == "tensor_map" and not slot.get("tensor_map") \
                and slot.get("implicit_base") is None:
            bound[name] = BoundInput(name, "tensor_map", ("tensor_map", bytes(128)))
    missing = sorted(
        name for name, slot in slots.items()
        if name not in bound and not slot.get("tensor_map") and slot.get("implicit_base") is None
    )
    if missing:
        raise MissingBindingsError(missing)
    return bound


_ELEM_BITS = {
    "PRED": 8, "U8": 8, "S8": 8, "E4M3": 8, "E5M2": 8, "UE8M0": 8, "UE4M3": 8, "UE5M3": 8,
    "U16": 16, "S16": 16, "F16": 16, "BF16": 16, "U32": 32, "S32": 32, "F32": 32, "TF32": 32,
    "U64": 64, "S64": 64, "F64": 64, "B128": 128, "E2M3": 6, "E3M2": 6, "S2F6": 6,
    "E2M1": 4, "U4": 4, "S4": 4,
}


def _bind_shape(slot: Mapping[str, Any], abi: tuple, source: BoundInput, bound: dict[str, BoundInput]) -> None:
    name = str(slot["name"])
    dims = list(slot["shape"])
    shape = source.shape or ()
    consts = [d.get("Const") if isinstance(d, Mapping) else None for d in dims]
    if len(dims) == len(shape):
        mismatch = [
            (axis, want, got) for axis, (want, got) in enumerate(zip(consts, shape))
            if want is not None and int(want) != int(got)
        ]
        if not mismatch:
            for axis, dim in enumerate(dims):
                if isinstance(dim, Mapping) and "Param" in dim:
                    target = abi[int(dim["Param"])]
                    target_name = str(target["name"])
                    value = int(shape[axis])
                    if target_name in bound:
                        given = bound[target_name]
                        if given.kind == "scalar" and given.native[1] != _scalar_bits(target_name, value, target.get("dtype")):
                            raise InputError(f"{target_name!r} disagrees with {name!r}.shape[{axis}]={value}")
                    else:
                        bound[target_name] = BoundInput(
                            target_name, "scalar", ("scalar", _scalar_bits(target_name, value, target.get("dtype")))
                        )
            return
    # Different rank or dims: accept a reinterpretation of the same bytes
    # (e.g. fp8 data in a uint8 slot) when the slot is fully static.
    if all(c is not None for c in consts):
        elem = str((slot.get("dtype") or {}).get("elem", "U8")).upper()
        lanes = int((slot.get("dtype") or {}).get("lanes", 1))
        bits = int(np.prod([int(c) for c in consts], dtype=np.int64)) * _ELEM_BITS.get(elem, 8) * lanes
        nbytes = len(source.native[1])
        if (bits + 7) // 8 == nbytes:
            return
    raise InputError(
        f"buffer {name!r}: array shape {tuple(shape)} ({source.dtype}) does not match the kernel's "
        f"declared shape {[c if c is not None else '?' for c in consts]} ({slot.get('dtype')})"
    )


def _memory_span(value: Any) -> tuple[int, int] | None:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return None  # torch tensors are copied to numpy; no host aliasing
    if not isinstance(value, np.ndarray) or value.nbytes == 0:
        return None
    lo, hi = np.lib.array_utils.byte_bounds(value)
    return lo, hi


def host_addresses(bound: Mapping[str, BoundInput]) -> dict[str, int]:
    return {name: b.host_addr for name, b in bound.items() if b.kind == "buffer" and b.host_addr is not None}


def _patch_descriptor_pointers(module: CompiledModule, bound: dict[str, BoundInput]) -> None:
    """Rewrite host descriptor images bound to plain buffers so their global
    address names the engine address of their base array."""

    pending = {name: b for name, b in bound.items() if b.patch_offset is not None}
    if not pending:
        return
    natives = {name: b.native for name, b in bound.items()}
    addresses = native().plan_global_addresses(module.handle, natives, host_addrs=host_addresses(bound))
    for name, b in pending.items():
        if b.base not in addresses:
            # The base array is not a kernel argument, and numsim-core only
            # allocates arguments some parameter references
            # (CONTRACT_REQUESTS W8-8).
            raise NotImplementedError(
                f"descriptor {name!r} is bound to a plain buffer parameter and addresses host array "
                f"{b.base!r}, which is not a kernel argument; numsim-core cannot place it yet (W8-8)"
            )
        image = bytearray(b.native[1])
        image[0:8] = int(addresses[b.base] + b.patch_offset).to_bytes(8, "little")
        bound[name] = replace(b, native=("buffer", bytes(image), None))


def _resolve_aliased_buffers(bound: dict[str, BoundInput], values: Mapping[str, Any]) -> None:
    """Host-input aliasing rule (dev-loop.md): buffer arguments bound to
    overlapping host memory share ONE engine allocation, so writes through
    one name are visible through the other and Racecheck sees the aliasing.

    Each connected group of overlapping (C-contiguous) host spans becomes one
    hidden region buffer ``__host_region_<i>`` holding the union of the
    bytes, and every member an ``ArgValue::View`` of it at its byte offset."""

    import ctypes

    spans: list[tuple[int, int, str]] = []
    for name, b in bound.items():
        if b.kind != "buffer" or name not in values:
            continue
        span = _memory_span(values[name])
        if span is not None:
            spans.append((span[0], span[1], name))
    spans.sort()
    groups: list[list[tuple[int, int, str]]] = []
    for lo, hi, name in spans:
        if groups and lo < max(h for _, h, _ in groups[-1]):
            groups[-1].append((lo, hi, name))
        else:
            groups.append([(lo, hi, name)])
    for index, group in enumerate(g for g in groups if len(g) > 1):
        for _, _, name in group:
            if not np.asarray(values[name]).flags.c_contiguous:
                raise NotImplementedError(
                    f"buffer argument {name!r} is a non-contiguous view aliasing another argument; "
                    "v2 binds aliased host memory as contiguous byte views"
                )
        lo = min(l for l, _, _ in group)
        hi = max(h for _, h, _ in group)
        region = f"__host_region_{index}"
        bound[region] = BoundInput(region, "buffer", ("buffer", ctypes.string_at(lo, hi - lo), None),
                                   dtype=np.dtype(np.uint8), shape=(hi - lo,), host_addr=lo)
        for start, end, name in group:
            b = bound[name]
            bound[name] = BoundInput(name, "view", ("view", region, start - lo, end - start),
                                     dtype=b.dtype, shape=b.shape, base=region)


def _select_outputs(bound: Mapping[str, BoundInput], outputs: Iterable[str] | Mapping[str, str] | None,
                    module: CompiledModule) -> list[tuple[str, str, str]]:
    """Return ``(canonical buffer, external output name, selector)`` triples."""

    buffers = {name for name, b in bound.items() if b.kind in ("buffer", "view") and not name.startswith("__")}
    if outputs is None:
        return [(name, name, name) for name in sorted(buffers)]
    if isinstance(outputs, str):
        raise TypeError("NumSim outputs must be an iterable of names, not a string")
    pairs = outputs.items() if isinstance(outputs, Mapping) else ((n, n) for n in outputs)
    aliases = {}
    for kernel in module.spec.kernels:
        for slot in kernel.host_abi:
            for alias in (slot["name"], slot.get("local_name"), *(slot.get("aliases") or ())):
                if alias:
                    aliases.setdefault(str(alias), str(slot["name"]))
                    aliases.setdefault(f"k{kernel.index}:{alias}", str(slot["name"]))
    selected: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for external, selector in pairs:
        canonical = aliases.get(selector, selector)
        if canonical in bound and bound[canonical].kind == "tensor_map" and bound[canonical].base is not None:
            canonical = bound[canonical].base  # a tensor map selects its base array
        if canonical not in buffers and not (canonical in bound and bound[canonical].kind in ("buffer", "view")):
            raise InputError(f"NumSim output selector {selector!r} does not identify a bound kernel buffer")
        if canonical in seen:
            raise InputError(f"NumSim buffer {canonical!r} is exposed as more than one output")
        seen.add(canonical)
        selected.append((canonical, external, selector))
    return selected


def _tensor_map_view(image: np.ndarray, base: np.ndarray, data: bytes) -> np.ndarray | None:
    """The logical tensor addressed by a ``numsim.cases.TensorMap`` image over
    ``base``: shape = global dims outermost first, strides from the image,
    dtype = the base array's (legacy returned outputs selected through a
    tensor map in this form)."""

    raw = np.ascontiguousarray(image).view(np.uint8).reshape(-1)
    rank = int(raw[59]) & 7
    if not 1 <= rank <= 5:
        return None
    dims = [int.from_bytes(raw[16 + 4 * a: 20 + 4 * a].tobytes(), "little") or 2**32 for a in range(rank)]
    strides_bytes = []
    for pair in range(2):
        packed = int.from_bytes(raw[36 + 9 * pair: 45 + 9 * pair].tobytes(), "little")
        strides_bytes += [(packed & ((1 << 36) - 1)) << 4, (packed >> 36) << 4]
    item = base.dtype.itemsize
    pointer = int.from_bytes(raw[0:8].tobytes(), "little")
    offset = pointer - int(base.__array_interface__["data"][0])
    flat = np.frombuffer(data, dtype=np.uint8)
    shape = tuple(reversed(dims))
    strides = tuple(reversed([item, *strides_bytes[: rank - 1]]))
    extent = offset + sum((d - 1) * st for d, st in zip(shape, strides)) + item
    if offset < 0 or extent > flat.size or offset % item:
        return None
    return np.lib.stride_tricks.as_strided(
        flat[offset:].view(base.dtype) if (flat.size - offset) % item == 0 else flat[offset: flat.size - (flat.size - offset) % item].view(base.dtype),
        shape=shape, strides=strides, writeable=False,
    )


def _input_digest(bound: Mapping[str, BoundInput]) -> str:
    digest = hashlib.sha256()
    for name in sorted(bound):
        digest.update(name.encode())
        digest.update(repr(bound[name].native[:1]).encode())
        for part in bound[name].native[1:]:
            digest.update(part if isinstance(part, bytes) else repr(part).encode())
    return digest.hexdigest()


class Engine:
    """Run v2 modules.

    ``max_workers`` (default 8, ``"auto"`` = detected CPU count) is the
    scheduler's worker-thread count (``RunConfig.workers``; results do not
    depend on it). ``native_loop_iteration_budget`` / ``native_loop_reschedule_quantum``
    map to the loop budget and slice quantum, ``backend`` selects
    ``"interp"`` or ``"codegen"`` (default ``NUMSIM_V2_BACKEND``)."""

    def __init__(
        self,
        max_workers: int | str = 8,
        *,
        native_loop_iteration_budget: int | None = None,
        native_loop_reschedule_quantum: int | None = None,
        seed: int | None = None,
        backend: str | None = None,
        opt_level: int = 1,
    ):
        opts = options()
        self.max_workers = (os.cpu_count() or 1) if max_workers == "auto" else int(max_workers)
        if self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer or 'auto'")
        self.loop_budget = native_loop_iteration_budget
        self.quantum = native_loop_reschedule_quantum
        self.seed = opts.seed if seed is None else int(seed)
        self.backend = backend or opts.backend
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.opt_level = opt_level
        self._phase_memo: dict[tuple, list[dict[str, Any]]] = {}

    # -- core call ---------------------------------------------------------
    def _native_run(self, module: CompiledModule, bound: Mapping[str, BoundInput], mode: str,
                    **extra: Any) -> dict[str, Any]:
        if "synccheck_limits" in extra:
            extra = {**extra, "synccheck_limits": dict(extra["synccheck_limits"])}
        try:
            return self._native_call(module, bound, mode, extra)
        except ValueError as error:
            if str(error).startswith("codegen backend:"):
                from tirx_harness.numsim.errors import NumSimBuildError

                raise NumSimBuildError(str(error)) from error
            raise

    def _native_call(self, module: CompiledModule, bound, mode: str, extra: Mapping[str, Any]) -> dict[str, Any]:
        return native().run(
            module.handle,
            {name: b.native for name, b in bound.items()},
            mode=mode,
            backend=self.backend,
            workers=self.max_workers,
            seed=self.seed,
            loop_budget=self.loop_budget,
            quantum=self.quantum,
            opt_level=self.opt_level,
            codegen_cache_dir=str(options().cache_root / "v2-codegen"),
            host_addrs=host_addresses(bound),
            **extra,
        )

    def address_of(self, module: CompiledModule, inputs: dict[str, Any], name: str) -> int:
        """Engine global address argument ``name`` (or a selector of it)
        will be bound at for these inputs (W8-7), e.g. to build raw-pointer
        input words before the run."""

        bound = canonicalize_inputs(module, inputs)
        natives = {key: b.native for key, b in bound.items()}
        addresses = native().plan_global_addresses(module.handle, natives, host_addrs=host_addresses(bound))
        lookup = {}
        for kernel in module.spec.kernels:
            for slot in kernel.host_abi:
                for alias in (slot["name"], slot.get("local_name"), *(slot.get("aliases") or ())):
                    if alias:
                        lookup.setdefault(str(alias), str(slot["name"]))
                        lookup.setdefault(f"k{kernel.index}:{alias}", str(slot["name"]))
        canonical = lookup.get(name, name)
        if canonical not in addresses:
            raise InputError(f"{name!r} is not a bound buffer argument")
        return int(addresses[canonical])

    @staticmethod
    def _subset_extra(subset: Any, assumptions: Any = None) -> dict[str, Any]:
        """``ExecutionSubset`` -> resident cluster ids for ``RunConfig::subset``."""

        # Host assumptions are dropped in v2 (W8-4 ruling): the only legacy
        # field, external_grid_dependencies_satisfied, is satisfied
        # automatically at a launch boundary, and no corpus case sets it.
        del assumptions
        if subset is None:
            return {}
        if getattr(subset, "cta_ids", None) is not None:
            raise NotImplementedError("v2 subsets select clusters; cta_ids subsets are not supported")
        clusters = getattr(subset, "cluster_ids", None)
        if clusters is None:
            return {}
        return {"subset": tuple(int(c) for c in clusters)}

    # -- NumSim ------------------------------------------------------------
    def run(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        subset: Any = None,
        assumptions: Any = None,
        outputs: Iterable[str] | Mapping[str, str] | None = None,
    ) -> NumSimResult:
        extra = self._subset_extra(subset, assumptions)
        started = time.perf_counter()
        bound = canonicalize_inputs(module, inputs)
        selected = _select_outputs(bound, outputs, module)
        bind_ms = (time.perf_counter() - started) * 1e3
        raw = self._native_run(module, bound, "numsim", **extra)
        report_started = time.perf_counter()
        result_outputs = {}
        for canonical, external, selector in selected:
            b = bound[canonical]
            data = raw["outputs"].get(canonical)
            if data is None and b.kind == "view":
                region = raw["outputs"].get(b.base)
                if region is not None:
                    _, _, start, length = b.native
                    data = region[start:start + length]
            if data is None:
                continue
            given = inputs.get(selector)
            if given is not None and getattr(given, "_tensor_map_base", None) is not None:
                view = _tensor_map_view(np.asarray(given), np.asarray(given._tensor_map_base), data)
                if view is not None:
                    result_outputs[external] = view
                    continue
            if isinstance(given, np.ndarray) and given.nbytes == len(data):
                # The selector's own array layout (two kernels may bind the
                # same bytes under one name with different dtypes).
                dtype, shape = given.dtype, given.shape
            else:
                dtype, shape = b.dtype, b.shape
            array = np.frombuffer(data, dtype=dtype)
            result_outputs[external] = array.reshape(shape) if shape is not None else array
        diagnostics = [
            diagnostic_from_core(
                d, _span_resolver(module, d["kernel_index"] if isinstance(d.get("kernel_index"), int) else None)
            )
            for d in raw["diagnostics"]
        ]
        attach_operations(diagnostics, 0, _site_info_resolver(module), _kernel_span_resolver(module))
        timing = _timing(module, bind_ms, raw, report_started)
        _raise_unless_completed(raw["status"], diagnostics)
        return NumSimResult(outputs=result_outputs, diagnostics=diagnostics, stats=dict(raw["stats"]),
                            status=dict(raw["status"]), timing=timing)

    # -- checkers ----------------------------------------------------------
    def _checker_run(self, module: CompiledModule, bound, mode: str, extra: Mapping[str, Any]) -> dict[str, Any]:
        """One checker run over every launch, memoized for the phase loop."""

        key = (mode, module.cache_key, _input_digest(bound), tuple(sorted(extra.items())))
        if key not in self._phase_memo:
            self._phase_memo.clear()  # keep at most one run alive
            self._phase_memo[key] = self._native_run(module, bound, mode, **extra)
        return self._phase_memo[key]

    def _checker_phase(self, mode: str, module: CompiledModule, inputs: dict[str, Any], phase_index: int,
                       extra: Mapping[str, Any]) -> AnalysisResult:
        kernels = module.spec.kernels
        if not 0 <= phase_index < len(kernels):
            raise ValueError(f"native {mode} phase {phase_index} is outside [0, {len(kernels)})")
        started = time.perf_counter()
        bound = canonicalize_inputs(module, inputs)
        bind_ms = (time.perf_counter() - started) * 1e3
        raw = self._checker_run(module, bound, mode, extra)
        report_started = time.perf_counter()
        span_of_kernel = _kernel_span_resolver(module)
        payloads = list(raw.get("payloads") or ())
        reports = [json.loads(r) for r in raw.get("reports") or ()]
        # Payloads and reports come one per launch, in launch order. A
        # payload's own `launch` field is used only when it is consistent
        # (a launch without sync events cannot name its kernel).
        numbers = [int(r.get("launch", i)) for i, r in enumerate(reports)] or list(range(len(payloads)))
        if len(set(numbers)) != len(numbers):
            numbers = list(range(len(payloads)))
        launches = dict(zip(numbers, payloads))
        base = launches.get(phase_index)
        if base is None:
            report = next((r for r in reports if int(r.get("launch", -1)) == phase_index), None)
            if report is not None:
                base = phase_payload(
                    checker=mode, phase_index=phase_index, phase_name=kernels[phase_index].name,
                    records=[record_from_core(f, lambda site: span_of_kernel(phase_index, site))
                             for f in report.get("findings", ())],
                    status={}, diagnostics=[], coverage=report.get("coverage", ()),
                )
            elif raw["status"].get("kind") not in (None, "completed"):
                # Only the engine's own stop (RunStatus) prevents later
                # launches; a checker verdict on an earlier launch never does.
                stop = next((d for d in raw["diagnostics"] if d.get("source") == "run_status"), {})
                failed = stop.get("kernel_index")
                cause = f"{stop.get('kind')}: {stop.get('message') or stop.get('reason') or ''}".strip(": ")
                base = {"verdict": "incomplete", "incomplete": [{
                    "kind": "analysis_incomplete", "status": "incomplete", "reason": "launch_not_executed",
                    "stopped_kernel_index": failed,
                    "message": (f"launch {phase_index} did not run: the engine stopped the module at "
                                f"launch {failed} ({cause})"),
                }]}
            else:
                base = {"verdict": "incomplete", "incomplete": [{
                    "kind": "analysis_incomplete", "status": "incomplete", "reason": "no_checker_report",
                    "message": f"the {mode} checker produced no report for launch {phase_index}",
                }]}
        last_launch = max(launches) if launches else len(reports) - 1
        diagnostics = []
        for diagnostic in raw["diagnostics"]:
            kernel = diagnostic.get("kernel_index")
            owner = kernel if isinstance(kernel, int) else last_launch
            if owner == phase_index:
                site = diagnostic.get("site")
                record = diagnostic_from_core(diagnostic, lambda s, k=owner: span_of_kernel(k, s))
                record.setdefault("source_span", span_of_kernel(owner, site) if isinstance(site, int) else None)
                diagnostics.append(record)
        payload = checker_phase_payload(
            base,
            checker=mode,
            phase_index=phase_index,
            phase_name=kernels[phase_index].name,
            status=raw["status"],
            diagnostics=diagnostics,
            span_of_kernel=span_of_kernel,
            site_info_of=_site_info_resolver(module),
        )
        payload.setdefault("stats", {}).update(raw.get("stats") or {})
        # One engine run serves every phase of a module; its build/run/check
        # times are repeated on each phase payload.
        payload["timing"] = _timing(module, bind_ms, raw, report_started)
        return AnalysisResult(mode, payload)

    def run_racecheck_phase(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        phase_index: int = 0,
        subset: Any = None,
        inspect_accesses: bool = False,
        max_polls: int | None = None,
        max_transitions: int | None = None,
        advance_prefix: bool = False,
    ) -> AnalysisResult:
        """Racecheck one launch phase. Earlier phases always execute first
        (v2 has no isolated-phase mode; ``advance_prefix`` is accepted)."""

        del inspect_accesses, max_polls, advance_prefix
        extra = self._subset_extra(subset)
        if max_transitions is not None:
            extra["max_rounds"] = int(max_transitions)
        return self._checker_phase("racecheck", module, inputs, phase_index, extra)

    def run_synccheck_phase(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        phase_index: int = 0,
        subset: Any = None,
        assumptions: Any = None,
        coverage_bounds: Any = None,
        resource_limits: Any = None,
        max_polls: int | None = None,
        max_transitions: int | None = None,
        advance_prefix: bool = False,
    ) -> AnalysisResult:
        del max_polls, advance_prefix
        extra = self._subset_extra(subset, assumptions)
        if resource_limits is not None:
            extra["state_budget"] = int(resource_limits.max_backtrack_nodes)
            extra["transition_budget"] = int(resource_limits.max_loop_steps)
            extra["synccheck_limits"] = tuple(sorted(
                (name, int(getattr(resource_limits, name)))
                for name in ("max_schedules", "max_events_per_run", "max_total_events",
                             "max_wall_time_ms", "max_diagnostic_bytes")
            ))
        if max_transitions is not None:
            extra["max_rounds"] = int(max_transitions)
        result = self._checker_phase("synccheck", module, inputs, phase_index, extra)
        coverage = result.payload.setdefault("coverage", {})
        if not isinstance(coverage, dict):
            coverage = result.payload["coverage"] = {}
        # The v2 explorer is not preemption-bounded: it explores every
        # interleaving of the projected protocol (sleep sets), so any
        # requested bound is satisfied. The requested bounds are echoed, not
        # dropped (numsim-behaviour-deltas, "Synccheck bounds and limits").
        if coverage_bounds is not None:
            coverage["requested_bounds"] = {
                "max_warp_preemptions": int(coverage_bounds.max_warp_preemptions),
                "max_completion_schedule_deviations": int(coverage_bounds.max_completion_schedule_deviations),
            }
        coverage["bounded_exploration"] = False
        return result


def _timing(module: CompiledModule, bind_ms: float, raw: Mapping[str, Any], report_started: float) -> dict[str, float]:
    engine = raw.get("timing") or {}
    return {
        "lower": float(module.lower_ms),
        "bind": bind_ms,
        "build": float(engine.get("build", 0.0)),
        "run": float(engine.get("run", 0.0)),
        "check": float(engine.get("check", 0.0)),
        "report": (time.perf_counter() - report_started) * 1e3,
    }


def _site_info_resolver(module: CompiledModule):
    kernels = module.spec.kernels

    def site_info_of(kernel: int, site: int) -> dict[str, Any] | None:
        if not 0 <= kernel < len(kernels) or not 0 <= site < len(kernels[kernel].sites):
            return None
        return kernels[kernel].sites[site]

    return site_info_of


def _kernel_span_resolver(module: CompiledModule):
    kernels = module.spec.kernels

    def span_of_kernel(kernel: int, site: int | None) -> dict[str, Any] | None:
        if site is None or not 0 <= kernel < len(kernels):
            return None
        return kernels[kernel].source_span(site)

    return span_of_kernel


def _span_resolver(module: CompiledModule, kernel_index: int | None):
    """Site resolver for one kernel; ``None`` = the diagnostic's own
    ``kernel_index`` is unknown, so use the first kernel that has the site."""

    kernels = module.spec.kernels
    by_kernel = _kernel_span_resolver(module)

    def span_of(site: int | None) -> dict[str, Any] | None:
        if kernel_index is not None:
            return by_kernel(kernel_index, site)
        for kernel in range(len(kernels)):
            span = by_kernel(kernel, site)
            if span is not None:
                return span
        return None

    return span_of


__all__ = ["BoundInput", "Engine", "ExecutionError", "InputError", "MissingBindingsError", "canonicalize_inputs"]
