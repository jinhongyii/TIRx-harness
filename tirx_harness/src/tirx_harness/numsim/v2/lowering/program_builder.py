"""Provisional Python mirror of the ``numsim-core`` ``Program`` contract.

This module owns the *shape* of the bytecode as the lowering sees it. The
authoritative definition is the Rust contract crate
``numsim/core-rs/numsim-core`` (``Program`` / ``Instr`` / ``SiteInfo`` ...),
which is not final yet. Every place where a Python field or variant must
match the contract is marked ``# CONTRACT:`` so the swap is mechanical: rename
the dataclass/field, adjust ``_variant_json``, and rerun the lowering tests.

JSON encoding follows serde's default *externally tagged* enum convention so
the Rust side can ``serde_json::from_str::<Program>`` without adapters:
unit variants are bare strings (``"Else"``), struct variants are
``{"Variant": {...fields}}``.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar


# --------------------------------------------------------------------------
# Scalar types and operands
# --------------------------------------------------------------------------

# CONTRACT: numsim_core::Dtype. Spelled as TVM dtype strings so the lowering
# never invents names; the contract enum must accept (or map) these.
SUPPORTED_DTYPES = frozenset(
    {
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "bfloat16", "float32", "float64",
        "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "float4_e2m1fn",
    }
)


class Space(str, enum.Enum):
    # CONTRACT: numsim_core::Space
    GLOBAL = "Global"
    SHARED = "Shared"          # shared::cta
    SHARED_CLUSTER = "SharedCluster"
    LOCAL = "Local"
    TMEM = "Tmem"
    GENERIC = "Generic"        # address must be resolved at run time


class SpecialReg(str, enum.Enum):
    # CONTRACT: numsim_core::Special — values the engine materializes.
    LANE = "Lane"                      # 0..32
    WARP_IN_CTA = "WarpInCta"
    THREAD_IN_CTA = "ThreadInCta"      # warp*32 + lane
    CTA_LINEAR = "CtaLinear"           # global linear CTA id
    CTA_IN_CLUSTER = "CtaInCluster"
    CLUSTER_LINEAR = "ClusterLinear"


@dataclass(frozen=True)
class Reg:
    """A ``[T; 32]`` register. ``dtype`` lives in ``Program.regs``."""

    index: int

    def to_json(self) -> Any:
        # CONTRACT: Operand::Reg(u32)
        return {"Reg": self.index}


@dataclass(frozen=True)
class Const:
    """Reference into ``Program.consts`` (broadcast to all lanes)."""

    index: int

    def to_json(self) -> Any:
        # CONTRACT: Operand::Const(u32)
        return {"Const": self.index}


Operand = Reg | Const


@dataclass(frozen=True)
class Scalar:
    # CONTRACT: numsim_core::Scalar { dtype, bits: u64 }. Floats are stored as
    # their IEEE bit pattern in ``dtype`` width so the Rust side never parses
    # decimal floats.
    dtype: str
    bits: int


@dataclass(frozen=True)
class RegDecl:
    # CONTRACT: Program.regs: Vec<RegDecl>  (MISSING from the plan sketch)
    dtype: str
    name: str = ""
    # Statically proven warp-uniform. A hint only: the engine must be correct
    # when it ignores it.
    uniform: bool = False


@dataclass(frozen=True)
class BufferDecl:
    # CONTRACT: Program.buffers: Vec<BufferDecl>  (MISSING from the plan sketch)
    name: str
    space: Space
    dtype: str
    shape: tuple[int, ...]
    # Element strides of the flat view; the lowering only emits element offsets.
    strides: tuple[int, ...]
    # Index into host_abi for parameter buffers, else None (kernel-allocated).
    param_slot: int | None = None
    # Byte size of one per-lane (Local) or per-CTA (Shared) backing.
    byte_size: int = 0
    alignment: int = 16


@dataclass(frozen=True)
class SourceSpan:
    file: str
    line: int
    column: int
    end_line: int
    end_column: int


@dataclass(frozen=True)
class SiteInfo:
    # CONTRACT: numsim_core::SiteInfo
    kind: str                       # TIRx node type key, e.g. "tirx.BufferStore"
    spans: tuple[SourceSpan, ...]   # >1 for SequentialSpan (macro/inline chains)
    op_name: str | None = None      # canonical op name for calls
    text: str = ""                  # short node rendering for reports
    dtype: str | None = None
    buffer: int | None = None       # index into Program.buffers when relevant


@dataclass(frozen=True)
class TileLayout:
    # CONTRACT: numsim_core::TileLayout. Element map table:
    # entries[lane][slot] = element offset into ``buffer`` (or -1 = none).
    buffer: int
    lanes: int
    slots: int
    entries: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Launch:
    # CONTRACT: numsim_core::Launch. Static ints today; see the design doc
    # (open question Q3) for scalar-parameter dependent extents.
    clusters: int
    ctas_per_cluster: int
    threads_per_cta: int
    warps_per_warpgroup: int = 4

    @property
    def warps_per_cta(self) -> int:
        return (self.threads_per_cta + 31) // 32


@dataclass(frozen=True)
class ParamSlot:
    # CONTRACT: numsim_core::ParamSlot. Kinds mirror legacy host_abi.py.
    name: str
    kind: str                         # "buffer" | "pointer" | "scalar" | "tensor_map"
    param_index: int
    dtype: str | None
    shape: tuple[int, ...] = ()


# --------------------------------------------------------------------------
# Instructions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Instr:
    """Base class. Subclasses are the ``Instr`` enum variants."""

    VARIANT: ClassVar[str] = ""

    def to_json(self) -> Any:
        return _variant_json(self)


# CONTRACT: every class below is one numsim_core::Instr variant. Field names
# are the serde field names.


@dataclass(frozen=True)
class Mov(Instr):
    VARIANT: ClassVar[str] = "Mov"
    dst: Reg
    src: Operand


@dataclass(frozen=True)
class ReadSpecial(Instr):
    VARIANT: ClassVar[str] = "ReadSpecial"
    dst: Reg
    special: SpecialReg


@dataclass(frozen=True)
class Binary(Instr):
    VARIANT: ClassVar[str] = "Binary"
    op: str           # Add Sub Mul Div Mod FloorDiv FloorMod Min Max And Or Xor Shl Shr
    dtype: str
    dst: Reg
    a: Operand
    b: Operand


@dataclass(frozen=True)
class Unary(Instr):
    VARIANT: ClassVar[str] = "Unary"
    op: str           # Neg Not BitNot
    dtype: str
    dst: Reg
    a: Operand


@dataclass(frozen=True)
class Compare(Instr):
    VARIANT: ClassVar[str] = "Compare"
    op: str           # Eq Ne Lt Le Gt Ge
    dtype: str        # operand dtype; result is bool
    dst: Reg
    a: Operand
    b: Operand


@dataclass(frozen=True)
class Cast(Instr):
    VARIANT: ClassVar[str] = "Cast"
    src_dtype: str
    dst_dtype: str
    dst: Reg
    a: Operand


@dataclass(frozen=True)
class Select(Instr):
    VARIANT: ClassVar[str] = "Select"
    dtype: str
    dst: Reg
    cond: Operand
    a: Operand
    b: Operand


@dataclass(frozen=True)
class Load(Instr):
    """Buffer-relative load: ``dst = buffers[buf][offset]`` (element offset)."""

    VARIANT: ClassVar[str] = "Load"
    dtype: str
    space: Space
    dst: Reg
    buf: int
    offset: Operand
    site: int


@dataclass(frozen=True)
class Store(Instr):
    VARIANT: ClassVar[str] = "Store"
    dtype: str
    space: Space
    buf: int
    offset: Operand
    value: Operand
    site: int


@dataclass(frozen=True)
class If(Instr):
    """Push mask; active &= cond. ``else_pc``/``end_pc`` let an all-false arm skip."""

    VARIANT: ClassVar[str] = "If"
    cond: Operand
    else_pc: int
    end_pc: int
    elect: bool = False   # CONTRACT: elect-gated region flag (legacy ElectSync)
    site: int = 0


@dataclass(frozen=True)
class Else(Instr):
    VARIANT: ClassVar[str] = "Else"
    end_pc: int


@dataclass(frozen=True)
class EndIf(Instr):
    VARIANT: ClassVar[str] = "EndIf"


@dataclass(frozen=True)
class LoopBegin(Instr):
    """Push loop frame (saved mask, break mask, iteration budget counter)."""

    VARIANT: ClassVar[str] = "LoopBegin"
    end_pc: int
    site: int


@dataclass(frozen=True)
class LoopIf(Instr):
    """Lanes with !cond leave the loop; if none remain, jump to ``end_pc``+1."""

    VARIANT: ClassVar[str] = "LoopIf"
    cond: Operand
    end_pc: int


@dataclass(frozen=True)
class LoopEnd(Instr):
    """Back-edge: restore continue-mask, count iteration, yield quantum, jump."""

    VARIANT: ClassVar[str] = "LoopEnd"
    head_pc: int


@dataclass(frozen=True)
class Exit(Instr):
    VARIANT: ClassVar[str] = "Exit"


@dataclass(frozen=True)
class Unsupported(Instr):
    """Reached at run time => verdict ``incomplete`` (fail closed)."""

    VARIANT: ClassVar[str] = "Unsupported"
    reason: str
    site: int


def _value_json(value: Any) -> Any:
    if isinstance(value, (Reg, Const, Instr)):
        return value.to_json()
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _value_json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_value_json(item) for item in value]
    return value


def _variant_json(instr: Instr) -> Any:
    body = {f.name: _value_json(getattr(instr, f.name)) for f in fields(instr)}
    if not body:
        return instr.VARIANT
    return {instr.VARIANT: body}


# --------------------------------------------------------------------------
# Program
# --------------------------------------------------------------------------


@dataclass
class Program:
    # CONTRACT: numsim_core::Program. ``regs``/``buffers``/``arch``/
    # ``unsupported`` are additions the plan sketch lacks.
    name: str
    code: list[Instr] = field(default_factory=list)
    consts: list[Scalar] = field(default_factory=list)
    sites: list[SiteInfo] = field(default_factory=list)
    layouts: list[TileLayout] = field(default_factory=list)
    topology: Launch | None = None
    host_abi: list[ParamSlot] = field(default_factory=list)
    regs: list[RegDecl] = field(default_factory=list)
    buffers: list[BufferDecl] = field(default_factory=list)
    arch: str | None = None
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _value_json(getattr(self, f.name)) for f in fields(self)}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


class ProgramBuilder:
    """Append-only builder with interning for consts and sites."""

    def __init__(self, name: str):
        self.program = Program(name=name)
        self._const_index: dict[tuple[str, int], int] = {}

    # -- tables ------------------------------------------------------------
    def reg(self, dtype: str, name: str = "", uniform: bool = False) -> Reg:
        self.program.regs.append(RegDecl(dtype=dtype, name=name, uniform=uniform))
        return Reg(len(self.program.regs) - 1)

    def reg_dtype(self, reg: Reg) -> str:
        return self.program.regs[reg.index].dtype

    def reg_uniform(self, reg: Reg) -> bool:
        return self.program.regs[reg.index].uniform

    def const(self, dtype: str, bits: int) -> Const:
        key = (dtype, bits)
        index = self._const_index.get(key)
        if index is None:
            self.program.consts.append(Scalar(dtype=dtype, bits=bits))
            index = len(self.program.consts) - 1
            self._const_index[key] = index
        return Const(index)

    def const_dtype(self, const: Const) -> str:
        return self.program.consts[const.index].dtype

    def site(self, info: SiteInfo) -> int:
        self.program.sites.append(info)
        return len(self.program.sites) - 1

    def buffer(self, decl: BufferDecl) -> int:
        self.program.buffers.append(decl)
        return len(self.program.buffers) - 1

    # -- code --------------------------------------------------------------
    @property
    def pc(self) -> int:
        return len(self.program.code)

    def emit(self, instr: Instr) -> int:
        self.program.code.append(instr)
        return len(self.program.code) - 1

    def patch(self, pc: int, instr: Instr) -> None:
        self.program.code[pc] = instr

    def finish(self) -> Program:
        if not self.program.code or not isinstance(self.program.code[-1], Exit):
            self.emit(Exit())
        return self.program
