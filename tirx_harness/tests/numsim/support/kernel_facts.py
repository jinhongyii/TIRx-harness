"""Static PrimFunc facts that corpus fixtures and their self-tests assert.

Plain TIRx reads (parameter buffer dtypes) plus the launch topology
of the lowered NumSim module (``numsim.v2.transpile``). Tests use these instead
of the legacy frontend's ``analyze`` / host ABI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def parameter_buffer_dtypes(kernel: Any) -> dict[str, str]:
    """Element dtype of every buffer or typed-pointer parameter of one PrimFunc."""

    from tvm.ir.type import PointerType, PrimType
    from tvm.tirx import BufferType

    dtypes: dict[str, str] = {}
    for parameter in kernel.params:
        ty = parameter.ty
        if isinstance(ty, BufferType):
            dtypes[str(parameter.name)] = str(parameter.dtype)
        elif isinstance(ty, PointerType) and isinstance(ty.element_type, PrimType):
            dtypes[str(parameter.name)] = str(ty.element_type.dtype)
    return dtypes


@dataclass(frozen=True)
class LaunchTopology:
    clusters: int
    ctas_per_cluster: int
    warps_per_cta: int

    @property
    def warp_count(self) -> int:
        return self.clusters * self.ctas_per_cluster * self.warps_per_cta


def _const(value: Any) -> int:
    if isinstance(value, dict):
        if "Const" not in value:
            raise ValueError(f"launch extent is not static: {value!r}")
        value = value["Const"]
    return int(value)


def launch_topology(kernel: Any) -> LaunchTopology:
    """Static launch shape of one PrimFunc, read from its lowered module."""

    from tirx_harness.numsim import v2

    (lowered,) = v2.transpile(kernel).document["kernels"]
    topology = lowered["topology"]
    ctas = math.prod(_const(extent) for extent in topology["grid"])
    ctas_per_cluster = math.prod(_const(extent) for extent in topology["cluster"])
    threads = math.prod(_const(extent) for extent in topology["block"])
    return LaunchTopology(
        clusters=ctas // ctas_per_cluster,
        ctas_per_cluster=ctas_per_cluster,
        warps_per_cta=threads // 32,
    )


def unsupported_reasons(kernel: Any) -> tuple[str, ...]:
    """Why NumSim cannot lower ``kernel`` (empty when it lowers cleanly)."""

    from tirx_harness.numsim import v2
    from tirx_harness.numsim.errors import UnsupportedTIRxError

    try:
        document = v2.transpile(kernel).document
    except UnsupportedTIRxError as error:
        return tuple(str(item) for item in (getattr(error, "unsupported", ()) or ())) or (str(error),)
    return tuple(str(item) for lowered in document["kernels"] for item in lowered.get("unsupported", ()))


__all__ = [
    "LaunchTopology",
    "launch_topology",
    "parameter_buffer_dtypes",
    "unsupported_reasons",
]
