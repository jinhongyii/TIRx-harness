"""Python emitter for the ``numsim_core::program`` contract (format version 1).

The authoritative types are in ``numsim/core-rs/numsim-core/src/program.rs``
(plus ``dtype.rs``, ``site.rs``, ``arena.rs``). This module mirrors them
closely enough that ``Module.to_json()`` deserializes with
``numsim_core::program::Module::from_json`` and passes ``Program::validate``.

Serde conventions (externally tagged): unit variants are strings, struct
variants are ``{"Variant": {...}}``, newtype structs (``Reg``, ``Buf``,
``Pc``, ``ConstId``, ...) are bare integers, ``Operand`` is
``{"Reg": n}``/``{"Const": n}``, ``Option`` is ``null``/value, tuples are
arrays, ``Box`` is transparent.

Instructions are generic :class:`Instr` objects (variant name + fields) checked
against :data:`SCHEMA`, one entry per ``Instr`` variant; the schema says how
each top-level field serializes. Nested contract structs (``MemMods``,
``BulkCompletion``, ``PhaseArg``, ...) are built with the helper functions
below, which return JSON-ready values (operands inside them are wrapped with
:func:`opnd`).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

FORMAT_VERSION = 3  # CONTRACT: numsim_core::program::FORMAT_VERSION (3: SiteInfo.buffers, W5-15)
SITE_NONE = 0xFFFF_FFFF

# ---------------------------------------------------------------------------
# Scalar types
# ---------------------------------------------------------------------------

# TVM dtype element name -> numsim_core::Dtype.
_DTYPES = {
    "bool": "Pred",
    "int8": "S8",
    "uint8": "U8",
    "int16": "S16",
    "uint16": "U16",
    "int32": "S32",
    "uint32": "U32",
    "int64": "S64",
    "uint64": "U64",
    "int128": "B128",
    "uint128": "B128",
    "float16": "F16",
    "bfloat16": "BF16",
    "float32": "F32",
    "float64": "F64",
    "float8_e4m3fn": "E4M3",
    "float8_e5m2": "E5M2",
    "float8_e8m0fnu": "UE8M0",
    "float6_e2m3fn": "E2M3",
    "float6_e3m2fn": "E3M2",
    "float4_e2m1fn": "E2M1",
    "int4": "S4",
    "uint4": "U4",
    "tf32": "TF32",
    "handle": "U64",
    "uint6": "U6",
    "float8_e3m4": "E3M4",
    "float8_e4m3": "E4M3Ieee",
    "float8_e4m3b11fnuz": "E4M3B11Fnuz",
    "float8_e4m3fnuz": "E4M3Fnuz",
    "float8_e5m2fnuz": "E5M2Fnuz",
}

_DTYPE_BITS = {
    "Pred": 8,
    "U8": 8,
    "S8": 8,
    "E4M3": 8,
    "E5M2": 8,
    "UE8M0": 8,
    "UE4M3": 8,
    "UE5M3": 8,
    "U16": 16,
    "S16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "S32": 32,
    "F32": 32,
    "TF32": 32,
    "U64": 64,
    "S64": 64,
    "F64": 64,
    "B128": 128,
    "E2M3": 6,
    "E3M2": 6,
    "S2F6": 6,
    "E2M1": 4,
    "U4": 4,
    "S4": 4,
    "U6": 6,
    "E3M4": 8,
    "E4M3Ieee": 8,
    "E4M3B11Fnuz": 8,
    "E4M3Fnuz": 8,
    "E5M2Fnuz": 8,
}

# PTX type token -> Dtype (b-types map to unsigned of the same width).
PTX_DTYPES = {
    "pred": "Pred",
    "b8": "U8",
    "u8": "U8",
    "s8": "S8",
    "b16": "U16",
    "u16": "U16",
    "s16": "S16",
    "b32": "U32",
    "u32": "U32",
    "s32": "S32",
    "b64": "U64",
    "u64": "U64",
    "s64": "S64",
    "b128": "B128",
    "f16": "F16",
    "bf16": "BF16",
    "f32": "F32",
    "f64": "F64",
    "tf32": "TF32",
    "e4m3": "E4M3",
    "e5m2": "E5M2",
    "ue8m0": "UE8M0",
    "e2m1": "E2M1",
    "e2m3": "E2M3",
    "e3m2": "E3M2",
    "ue4m3": "UE4M3",
    "ue5m3": "UE5M3",
}


class UnrepresentableType(ValueError):
    pass


@dataclass(frozen=True)
class Ty:
    """``numsim_core::Ty { elem: Dtype, lanes: u8 }``."""

    elem: str
    lanes: int = 1

    @staticmethod
    def from_tvm(name: str) -> Ty:
        if name in _DTYPES:
            return Ty(_DTYPES[name])
        base, sep, lanes = name.rpartition("x")
        if sep and lanes.isdigit() and base in _DTYPES:
            ty = Ty(_DTYPES[base], int(lanes))
            if ty.bits <= 256:
                return ty
        raise UnrepresentableType(f"dtype {name!r} has no numsim_core::Ty")

    @staticmethod
    def from_ptx(token: str, lanes: int = 1) -> Ty:
        base, sep, count = token.rpartition("x")
        if sep and count.isdigit() and base in PTX_DTYPES:
            return Ty(PTX_DTYPES[base], int(count) * lanes)
        if token not in PTX_DTYPES:
            raise UnrepresentableType(f"PTX type {token!r} has no numsim_core::Dtype")
        return Ty(PTX_DTYPES[token], lanes)

    @property
    def bits(self) -> int:
        return _DTYPE_BITS[self.elem] * self.lanes

    def with_lanes(self, lanes: int) -> Ty:
        return Ty(self.elem, lanes)

    def to_json(self) -> Any:
        return {"elem": self.elem, "lanes": self.lanes}


# ---------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reg:
    index: int

    def to_json(self) -> Any:  # bare newtype (Reg-typed fields)
        return self.index


@dataclass(frozen=True)
class Const:
    index: int


Operand = Reg | Const


def opnd(value: Operand) -> Any:
    """JSON of an ``Operand``."""
    if isinstance(value, Reg):
        return {"Reg": value.index}
    if isinstance(value, Const):
        return {"Const": value.index}
    raise TypeError(f"not an operand: {value!r}")


def opt_opnd(value: Operand | None) -> Any:
    return None if value is None else opnd(value)


# ---------------------------------------------------------------------------
# Nested contract values (JSON builders)
# ---------------------------------------------------------------------------


def mem_mods(
    cache: str = "Default",
    evict: str = "Normal",
    l2_prefetch: int = 0,
    policy: Operand | None = None,
    nc: bool = False,
    uniform: bool = False,
) -> dict[str, Any]:
    return {
        "cache": cache,
        "evict": evict,
        "l2_prefetch": l2_prefetch,
        "policy": opt_opnd(policy),
        "nc": nc,
        "uniform": uniform,
    }


def bulk_completion(mbar: Operand | None, space: str = "Shared") -> Any:
    if mbar is None:
        return "Group"
    return {"Mbarrier": {"mbar": opnd(mbar), "space": space}}


def phase_parity(value: Operand) -> Any:
    return {"Parity": opnd(value)}


def phase_state(value: Operand) -> Any:
    return {"State": opnd(value)}


# ---------------------------------------------------------------------------
# DimExpr
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimExpr:
    op: str  # Const Param Add Sub Mul FloorDiv CeilDiv Min Max
    value: int = 0
    args: tuple[DimExpr, ...] = ()

    @staticmethod
    def const(value: int) -> DimExpr:
        return DimExpr("Const", int(value))

    @staticmethod
    def param(slot: int) -> DimExpr:
        return DimExpr("Param", int(slot))

    @property
    def is_const(self) -> bool:
        return self.op == "Const"

    def to_json(self) -> Any:
        if self.op in ("Const", "Param"):
            return {self.op: self.value}
        return {self.op: [self.args[0].to_json(), self.args[1].to_json()]}


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

# Field kinds: reg, opt_reg, regs, op, opt_op, ops, pc, ty, int, bool, str_id,
# opt_str_id, buf, param, pred, opid, json (pre-built JSON; Reg -> int).
SCHEMA: dict[str, dict[str, str] | None] = {
    # control
    "Nop": None,
    "EndIf": None,
    "Break": None,
    "Continue": None,
    "Exit": None,
    "GridSync": None,
    "If": {"cond": "op", "else_pc": "pc", "end_pc": "pc", "elect": "bool"},
    "Else": {"end_pc": "pc"},
    "LoopBegin": {"end_pc": "pc"},
    "LoopIf": {"cond": "op", "end_pc": "pc"},
    "LoopEnd": {"head_pc": "pc"},
    "Assert": {"cond": "op", "msg": "opt_str_id"},
    "Unsupported": {"reason": "str_id"},
    # registers / TIR arithmetic
    "Mov": {"dst": "reg", "src": "op"},
    "ReadSpecial": {"dst": "reg", "sreg": "json"},
    "ReadParam": {"dst": "reg", "slot": "param"},
    "Unary": {"op": "json", "ty": "ty", "dst": "reg", "a": "op"},
    "Binary": {"op": "json", "ty": "ty", "dst": "reg", "a": "op", "b": "op"},
    "Ternary": {"op": "json", "ty": "ty", "dst": "reg", "a": "op", "b": "op", "c": "op"},
    "Compare": {"op": "json", "ty": "ty", "dst": "reg", "a": "op", "b": "op"},
    "Select": {"ty": "ty", "dst": "reg", "cond": "op", "a": "op", "b": "op"},
    "Cast": {"from": "ty", "to": "ty", "dst": "reg", "src": "op", "rnd": "json", "sat": "bool"},
    "Ptx": {"op": "opid", "dsts": "regs", "srcs": "ops", "pred": "opt_op", "keep_dst": "bool"},
    "LoadRegIndexed": {"dst": "reg", "base": "reg", "len": "int", "idx": "op"},
    "StoreRegIndexed": {"base": "reg", "len": "int", "idx": "op", "value": "op"},
    # warp collectives
    "Shfl": {
        "mode": "json",
        "ty": "ty",
        "dst": "reg",
        "dst_pred": "opt_reg",
        "src": "op",
        "lane": "op",
        "clamp": "op",
        "membermask": "op",
    },
    "Vote": {"mode": "json", "dst": "reg", "pred": "op", "membermask": "op"},
    "Redux": {"op": "json", "ty": "ty", "dst": "reg", "src": "op", "membermask": "op"},
    "Elect": {"dst_pred": "reg", "dst_lane": "opt_reg", "membermask": "op"},
    "WarpSync": {"membermask": "op"},
    "LdMatrix": {
        "dsts": "regs",
        "addr": "op",
        "space": "json",
        "shape": "json",
        "num": "int",
        "trans": "bool",
        "fmt": "json",
    },
    "StMatrix": {
        "srcs": "ops",
        "addr": "op",
        "space": "json",
        "shape": "json",
        "num": "int",
        "trans": "bool",
    },
    # memory
    "Load": {
        "ty": "ty",
        "dst": "reg",
        "buf": "buf",
        "offset": "op",
        "sem": "json",
        "scope": "json",
        "mods": "json",
    },
    "Store": {
        "ty": "ty",
        "buf": "buf",
        "offset": "op",
        "value": "op",
        "sem": "json",
        "scope": "json",
        "mods": "json",
    },
    "LoadAddr": {
        "ty": "ty",
        "dst": "reg",
        "addr": "op",
        "space": "json",
        "sem": "json",
        "scope": "json",
        "mods": "json",
    },
    "StoreAddr": {
        "ty": "ty",
        "addr": "op",
        "space": "json",
        "value": "op",
        "sem": "json",
        "scope": "json",
        "mods": "json",
    },
    "AddrOf": {"dst": "reg", "buf": "buf", "offset": "op"},
    "Atom": {
        "op": "json",
        "ty": "ty",
        "dst": "opt_reg",
        "addr": "op",
        "space": "json",
        "value": "op",
        "cmp": "opt_op",
        "sem": "json",
        "scope": "json",
        "ftz": "bool",
    },
    "StBulk": {"addr": "op", "space": "json", "size": "op"},
    "Discard": {"addr": "op", "space": "json", "size": "int"},
    "Cvta": {"dst": "reg", "src": "op", "space": "json", "to_generic": "bool"},
    "Isspacep": {"dst": "reg", "src": "op", "space": "json"},
    "Mapa": {"dst": "reg", "src": "op", "rank": "op", "space": "json"},
    "GetCtaRank": {"dst": "reg", "src": "op", "space": "json"},
    # async copies
    "CpAsync": {
        "dst": "op",
        "src": "op",
        "cp_size": "int",
        "src_size": "opt_op",
        "ignore_src": "opt_op",
        "mods": "json",
    },
    "AsyncCommit": {"domain": "json"},
    "AsyncWait": {"domain": "json", "n": "int", "read": "bool"},
    "CpAsyncMbarArrive": {"mbar": "op", "space": "json", "noinc": "bool"},
    "BulkCopy": {
        "dst": "op",
        "dst_space": "json",
        "src": "op",
        "src_space": "json",
        "size": "op",
        "completion": "json",
        "multicast": "opt_op",
        "reduce": "json",
        "byte_mask": "opt_op",
        "ignore_oob": "json",
        "report": "json",
        "mods": "json",
    },
    "Tma": {
        "dir": "json",
        "mode": "json",
        "tmap": "op",
        "tmap_space": "json",
        "coords": "ops",
        "im2col_offsets": "ops",
        "smem": "op",
        "smem_space": "json",
        "completion": "json",
        "multicast": "opt_op",
        "cta_group": "int",
        "overrides": "json",
        "report": "json",
        "mods": "json",
    },
    "StAsync": {
        "ty": "ty",
        "value": "op",
        "addr": "op",
        "mbar": "opt_op",
        "red": "json",
        "sem": "json",
        "scope": "json",
    },
    "TensorMapReplace": {
        "tmap": "op",
        "space": "json",
        "field": "json",
        "ord": "json",
        "value": "op",
    },
    "TensorMapCopyFence": {"dst": "op", "src": "op", "size": "int", "scope": "json"},
    # synchronization
    "Barrier": {"kind": "json", "id": "op", "count": "opt_op", "aligned": "bool"},
    "ClusterArrive": {"sem": "json", "aligned": "bool"},
    "ClusterWait": {"acquire": "bool", "aligned": "bool"},
    "MbarInit": {"mbar": "op", "space": "json", "count": "op", "layout_v1": "bool"},
    "MbarInval": {"mbar": "op", "space": "json"},
    "MbarArrive": {
        "mbar": "op",
        "space": "json",
        "count": "opt_op",
        "expect_tx": "opt_op",
        "drop": "bool",
        "no_complete": "bool",
        "sem": "json",
        "scope": "json",
        "multicast": "opt_op",
        "state": "opt_reg",
    },
    "MbarTx": {
        "op": "json",
        "mbar": "op",
        "space": "json",
        "bytes": "op",
        "multicast": "opt_op",
        "scope": "json",
    },
    "MbarTestWait": {
        "kind": "json",
        "mbar": "op",
        "space": "json",
        "phase": "json",
        "sem": "json",
        "scope": "json",
        "dst": "opt_reg",
        "report": "opt_reg",
        "report_value": "opt_reg",
    },
    "MbarWait": {"mbar": "op", "space": "json", "phase": "json", "sem": "json", "scope": "json"},
    "MbarQuery": {"dst": "reg", "op": "json"},
    "Fence": {"kind": "json", "sem": "json", "scope": "json"},
    "SetMaxNReg": {"inc": "bool", "count": "int"},
    "WaitUntil": {
        "dst": "reg",
        "addr": "op",
        "ty": "ty",
        "space": "json",
        "sem": "json",
        "scope": "json",
        "pred": "pred",
        "captures": "regs",
    },
    "GridDepControl": {"launch_dependents": "bool"},
    "ClcTryCancel": {"resp": "op", "mbar": "op", "multicast": "bool"},
    # tcgen05
    "TcgenAlloc": {"dst": "op", "ncols": "op", "cta_group": "int", "exclusive": "bool"},
    "TcgenDealloc": {"taddr": "op", "ncols": "op", "cta_group": "int", "exclusive": "bool"},
    "TcgenRelinquish": {"cta_group": "int"},
    "TcgenCommit": {
        "mbar": "op",
        "space": "json",
        "cta_group": "int",
        "multicast": "opt_op",
        "sync_restrict": "bool",
        "multicast_width": "json",
    },
    "TcgenLd": {
        "dsts": "regs",
        "taddr": "op",
        "row": "op",
        "col": "op",
        "shape": "json",
        "num": "int",
        "pack": "bool",
        "red": "json",
        "red_abs": "bool",
        "red_nan": "bool",
        "spcompress": "bool",
    },
    "TcgenSt": {
        "srcs": "ops",
        "taddr": "op",
        "row": "op",
        "col": "op",
        "shape": "json",
        "num": "int",
        "unpack": "bool",
    },
    "TcgenWait": {"st": "bool"},
    "TcgenCp": {
        "taddr": "op",
        "row": "op",
        "col": "op",
        "sdesc": "op",
        "rows": "int",
        "bits": "int",
        "multicast": "int",
        "decompress_bits": "int",
        "cta_group": "int",
    },
    "TcgenMma": {
        "kind": "json",
        "cta_group": "int",
        "d": "op",
        "a": "json",
        "b_desc": "op",
        "idesc": "op",
        "enable_input_d": "op",
        "ws": "bool",
        "ws_b_buffer": "int",
        "block_scale": "json",
        "scale_input_d": "opt_op",
        "sparse_meta": "opt_op",
        "disable_output_lane": "ops",
        "collector_a": "json",
        "collector_b": "json",
        "ashift": "bool",
        "lut_b": "bool",
        "lut_b_addr": "opt_op",
        "declared": "json",
    },
}

# Variants that may return Blocked (mirror of Instr::may_block).
BLOCKING = frozenset(
    {
        "ClusterWait",
        "GridSync",
        "MbarWait",
        "AsyncWait",
        "TcgenAlloc",
        "TcgenWait",
        "WaitUntil",
        "WarpSync",
    }
)

# Instructions allowed inside a wait_until predicate range.
PRED_ALLOWED = frozenset(
    {
        "Mov",
        "Unary",
        "Binary",
        "Ternary",
        "Compare",
        "Select",
        "Cast",
        "Ptx",
        "Load",
        "LoadAddr",
        "LoadRegIndexed",
    }
)


class Instr:
    """One ``numsim_core::Instr`` (variant + fields, checked against SCHEMA)."""

    __slots__ = ("fields", "variant")

    def __init__(self, variant: str, /, **fields: Any):
        if variant not in SCHEMA:
            raise KeyError(f"unknown Instr variant {variant!r}")
        schema = SCHEMA[variant] or {}
        missing = set(schema) - set(fields)
        extra = set(fields) - set(schema)
        if missing or extra:
            raise TypeError(f"{variant}: missing {sorted(missing)} extra {sorted(extra)}")
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "fields", fields)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Instr is immutable")

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Instr)
            and self.variant == other.variant
            and self.fields == other.fields
        )

    def __hash__(self) -> int:
        return hash((self.variant, tuple(sorted(self.fields))))

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.fields.items())
        return f"{self.variant}({body})"

    @property
    def may_block(self) -> bool:
        if self.variant == "Barrier":
            return bool(self.fields["kind"] != "Arrive")
        if self.variant == "SetMaxNReg":
            return True
        if self.variant in ("TcgenDealloc", "TcgenRelinquish"):
            return bool(self.fields["cta_group"] == 2)
        return self.variant in BLOCKING

    def operands(self) -> list[Operand]:
        """Every operand/register read by this instruction (top-level fields)."""
        out: list[Operand] = []
        for name, kind in (SCHEMA[self.variant] or {}).items():
            value = self.fields[name]
            if kind in ("op", "opt_op") and value is not None:
                out.append(value)
            elif kind == "ops":
                out.extend(value)
            elif (
                kind == "reg"
                and self.variant in ("LoadRegIndexed", "StoreRegIndexed")
                and name == "base"
            ):
                out.append(value)
        return out

    def writes(self) -> list[Reg]:
        out: list[Reg] = []
        for name, kind in (SCHEMA[self.variant] or {}).items():
            value = self.fields[name]
            if kind in ("reg", "opt_reg") and value is not None and name not in ("base",):
                out.append(value)
            elif kind == "regs" and name == "dsts":
                out.extend(value)
        return out

    def to_json(self) -> Any:
        schema = SCHEMA[self.variant]
        if schema is None:
            return self.variant
        return {
            self.variant: {
                name: _field_json(kind, self.fields[name]) for name, kind in schema.items()
            }
        }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Reg):
        return value.index
    if isinstance(value, Const):
        return {"Const": value.index}
    if isinstance(value, (Ty, DimExpr)):
        return value.to_json()
    if isinstance(value, dict):
        return {k: _plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(v) for v in value]
    return value


def _field_json(kind: str, value: Any) -> Any:
    if kind in ("op",):
        return opnd(value)
    if kind == "opt_op":
        return opt_opnd(value)
    if kind == "ops":
        return [opnd(v) for v in value]
    if kind in ("reg", "opt_reg"):
        return None if value is None else value.index
    if kind == "regs":
        return [v.index for v in value]
    if kind == "ty":
        return value.to_json()
    if kind in ("pc", "int", "buf", "param", "pred", "opid", "str_id", "opt_str_id", "bool"):
        return value
    if kind == "json":
        return _plain_json(value)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpan:
    file: str | None
    line: int
    column: int
    end_line: int
    end_column: int

    def to_json(self) -> Any:
        return {
            "file": self.file,
            "line": self.line,
            "col": self.column,
            "end_line": self.end_line,
            "end_col": self.end_column,
        }


@dataclass(frozen=True)
class SiteInfo:
    kind: str
    spans: tuple[SourceSpan, ...]
    op_name: str = ""
    text: str = ""
    dtype: str | None = None
    buffer: str | None = None
    # W5-15: logical buffer of each pointer operand, in operand order (None = raw
    # pointer). Empty: derived as `[buffer]`. `buffer` stays `buffers[0]`.
    buffers: tuple[str | None, ...] = ()

    def all_buffers(self) -> list[str | None]:
        if self.buffers:
            return list(self.buffers)
        return [self.buffer] if self.buffer is not None else []

    def to_json(self) -> Any:
        buffers = self.all_buffers()
        return {
            "kind": self.kind,
            "spans": [s.to_json() for s in self.spans],
            "op_name": self.op_name,
            "text": self.text,
            "dtype": self.dtype,
            "buffer": buffers[0] if buffers else None,
            "buffers": buffers,
        }


@dataclass(frozen=True)
class RegDecl:
    ty: Ty
    name: str | None = None
    uniform: bool = False

    def to_json(self) -> Any:
        return {"ty": self.ty.to_json(), "name": self.name, "uniform": self.uniform}


@dataclass(frozen=True)
class BufferDecl:
    name: str
    space: str  # Global Shared Local Param Tmem Reg
    dtype: Ty
    shape: tuple[DimExpr, ...]
    strides: tuple[DimExpr, ...] = ()
    param_slot: int | None = None
    base: int = 0
    byte_len: DimExpr | None = None
    align: int = 16
    view_of: int | None = None
    sync_words: bool = False
    base_reg: Reg | None = None  # runtime TMEM base (contract item 22); base must be 0

    def to_json(self) -> Any:
        return {
            # arena::Space serializes snake_case (contract e2551cf).
            "name": self.name,
            "space": self.space.lower(),
            "dtype": self.dtype.to_json(),
            "shape": [d.to_json() for d in self.shape],
            "strides": [d.to_json() for d in self.strides],
            "param_slot": self.param_slot,
            "base": self.base,
            "byte_len": None if self.byte_len is None else self.byte_len.to_json(),
            "align": self.align,
            "view_of": self.view_of,
            "sync_words": self.sync_words,
            "base_reg": None if self.base_reg is None else self.base_reg.index,
        }


@dataclass(frozen=True)
class OpKey:
    name: str
    mods: tuple[str, ...] = ()

    def to_json(self) -> Any:
        return {"name": self.name, "mods": list(self.mods)}


@dataclass(frozen=True)
class PredProgram:
    arg: Reg
    start: int
    end: int
    result: Reg
    reads_memory: bool

    def to_json(self) -> Any:
        return {
            "arg": self.arg.index,
            "start": self.start,
            "end": self.end,
            "result": self.result.index,
            "reads_memory": self.reads_memory,
        }


@dataclass
class Launch:
    grid: tuple[DimExpr, DimExpr, DimExpr]
    cluster: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (32, 1, 1)
    static_smem_bytes: int = 0
    dyn_smem_bytes: DimExpr = field(default_factory=lambda: DimExpr.const(0))
    min_blocks_per_sm: int | None = None
    cooperative: bool = False
    regs_per_thread: int = 0

    @property
    def threads_per_cta(self) -> int:
        return self.block[0] * self.block[1] * self.block[2]

    @property
    def warps_per_cta(self) -> int:
        return (self.threads_per_cta + 31) // 32

    def to_json(self) -> Any:
        return {
            "grid": [d.to_json() for d in self.grid],
            "cluster": list(self.cluster),
            "block": list(self.block),
            "static_smem_bytes": self.static_smem_bytes,
            "dyn_smem_bytes": self.dyn_smem_bytes.to_json(),
            "min_blocks_per_sm": self.min_blocks_per_sm,
            "cooperative": self.cooperative,
            "regs_per_thread": self.regs_per_thread,
        }


@dataclass(frozen=True)
class TensorMapSpec:
    dtype: str  # numsim_core::Dtype name
    rank: int
    global_dim: tuple[DimExpr, ...]
    global_stride: tuple[DimExpr, ...]
    box_dim: tuple[DimExpr, ...]  # contract item 30: DimExpr (runtime prologue values)
    element_stride: tuple[DimExpr, ...]
    interleave: int
    swizzle: int
    l2_promotion: int
    oob_fill: int
    base_offset: DimExpr
    force_cu_dtype: int | None = None  # raw CUtensorMapDataType when it differs from dtype

    def to_json(self) -> Any:
        return {
            "dtype": self.dtype,
            "rank": self.rank,
            "global_dim": [d.to_json() for d in self.global_dim],
            "global_stride": [d.to_json() for d in self.global_stride],
            "box_dim": [d.to_json() for d in self.box_dim],
            "element_stride": [d.to_json() for d in self.element_stride],
            "interleave": self.interleave,
            "swizzle": self.swizzle,
            "l2_promotion": self.l2_promotion,
            "oob_fill": self.oob_fill,
            "base_offset": self.base_offset.to_json(),
            "force_cu_dtype": self.force_cu_dtype,
        }


@dataclass
class ParamSlot:
    name: str
    kind: str  # Buffer Pointer Scalar TensorMap ImplicitShape
    dtype: Ty | None = None
    shape: tuple[DimExpr, ...] = ()
    tensor_map: TensorMapSpec | None = None
    implicit_base: int | None = None
    buf: int | None = None
    local_name: str = ""
    aliases: tuple[str, ...] = ()
    # Lowering-side facts not in the contract (kept for tests/diagnostics).
    param_index: int = -1
    shape_of: tuple[int, int] | None = None

    def to_json(self) -> Any:
        return {
            "name": self.name,
            "local_name": self.local_name or self.name,
            "aliases": list(self.aliases),
            "kind": self.kind
            if self.shape_of is None
            else {"ImplicitShape": {"buffer": self.shape_of[0], "axis": self.shape_of[1]}},
            "dtype": None if self.dtype is None else self.dtype.to_json(),
            "shape": [d.to_json() for d in self.shape],
            "tensor_map": None if self.tensor_map is None else self.tensor_map.to_json(),
            "implicit_base": self.implicit_base,
            "buf": self.buf,
        }


@dataclass
class Requirements:
    implicit_tmem: bool = False
    dynamic_tmem_lifecycle: bool = False
    readonly_proxy: bool = False
    grid_dependency: bool = False
    raw_tensor_map_registry: bool = False

    def to_json(self) -> Any:
        return dict(self.__dict__)


@dataclass
class Program:
    name: str
    code: list[Instr] = field(default_factory=list)
    code_sites: list[int] = field(default_factory=list)
    consts: list[tuple[Ty, int]] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)
    sites: list[SiteInfo] = field(default_factory=list)
    regs: list[RegDecl] = field(default_factory=list)
    buffers: list[BufferDecl] = field(default_factory=list)
    ops: list[OpKey] = field(default_factory=list)
    preds: list[PredProgram] = field(default_factory=list)
    layouts: list[Any] = field(default_factory=list)
    topology: Launch | None = None
    host_abi: list[ParamSlot] = field(default_factory=list)
    arch: str | None = None
    requirements: Requirements = field(default_factory=Requirements)
    unsupported: list[str] = field(default_factory=list)

    def site_of(self, pc: int) -> SiteInfo | None:
        index = self.code_sites[pc]
        return None if index == SITE_NONE else self.sites[index]

    def const_value(self, operand: Operand) -> int:
        assert isinstance(operand, Const)
        return self.consts[operand.index][1]

    def to_dict(self) -> dict[str, Any]:
        topology = self.topology or Launch(grid=(DimExpr.const(1),) * 3)
        return {
            "name": self.name,
            "code": [i.to_json() for i in self.code],
            "code_sites": list(self.code_sites),
            "consts": [{"ty": ty.to_json(), "bits": bits} for ty, bits in self.consts],
            "strings": list(self.strings),
            "sites": [s.to_json() for s in self.sites],
            "regs": [r.to_json() for r in self.regs],
            "buffers": [b.to_json() for b in self.buffers],
            "ops": [o.to_json() for o in self.ops],
            "preds": [p.to_json() for p in self.preds],
            "layouts": [],
            "topology": topology.to_json(),
            "host_abi": [p.to_json() for p in self.host_abi],
            "arch": self.arch,
            "requirements": self.requirements.to_json(),
            "unsupported": list(self.unsupported),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class Module:
    kernels: list[Program]

    def to_dict(self) -> dict[str, Any]:
        return {"format_version": FORMAT_VERSION, "kernels": [k.to_dict() for k in self.kernels]}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ProgramBuilder:
    """Append-only builder with interning for consts, strings, sites and ops."""

    def __init__(self, name: str):
        self.program = Program(name=name)
        self._const_index: dict[tuple[Ty, int], int] = {}
        self._op_index: dict[OpKey, int] = {}
        self._site_index: dict[Any, int] = {}
        self._string_index: dict[str, int] = {}
        self.code: list[Instr] = self.program.code
        self.code_sites: list[int] = self.program.code_sites

    # -- tables ------------------------------------------------------------
    def reg(self, ty: Ty, name: str | None = None, uniform: bool = False) -> Reg:
        self.program.regs.append(RegDecl(ty=ty, name=name or None, uniform=uniform))
        return Reg(len(self.program.regs) - 1)

    def reg_ty(self, reg: Reg) -> Ty:
        return self.program.regs[reg.index].ty

    def reg_uniform(self, reg: Reg) -> bool:
        return self.program.regs[reg.index].uniform

    def set_uniform(self, reg: Reg, uniform: bool) -> None:
        decl = self.program.regs[reg.index]
        if decl.uniform != uniform:
            self.program.regs[reg.index] = RegDecl(decl.ty, decl.name, uniform)

    def const(self, ty: Ty, bits: int) -> Const:
        bits &= (1 << 128) - 1
        key = (ty, bits)
        index = self._const_index.get(key)
        if index is None:
            self.program.consts.append(key)
            index = len(self.program.consts) - 1
            self._const_index[key] = index
        return Const(index)

    def const_ty(self, const: Const) -> Ty:
        return self.program.consts[const.index][0]

    def string(self, text: str) -> int:
        index = self._string_index.get(text)
        if index is None:
            self.program.strings.append(text)
            index = len(self.program.strings) - 1
            self._string_index[text] = index
        return index

    def site(self, info: SiteInfo, key: Any = None) -> int:
        if key is not None:
            index = self._site_index.get(key)
            if index is not None:
                return index
        self.program.sites.append(info)
        index = len(self.program.sites) - 1
        if key is not None:
            self._site_index[key] = index
        return index

    def op(self, key: OpKey) -> int:
        index = self._op_index.get(key)
        if index is None:
            self.program.ops.append(key)
            index = len(self.program.ops) - 1
            self._op_index[key] = index
        return index

    def buffer(self, decl: BufferDecl) -> int:
        if not decl.name:
            # Every BufferDecl is named (W5-9): unnamed TIR buffers get a
            # stable synthetic name (view: parent + byte base + dtype).
            buffers = self.program.buffers
            if decl.view_of is not None:
                ty = decl.dtype.elem.lower() + (
                    "" if decl.dtype.lanes == 1 else f"x{decl.dtype.lanes}"
                )
                name = f"{buffers[decl.view_of].name}+{decl.base}.{ty}"
                if any(b.name == name for b in buffers):
                    # Distinct unnamed DeclBuffers stay distinct identities
                    # (legacy: one per view object; racecheck alias advisories).
                    name = f"{name}#{len(buffers)}"
                decl = dataclasses.replace(decl, name=name)
            else:
                decl = dataclasses.replace(decl, name=f"{decl.space.lower()}{len(buffers)}")
        self.program.buffers.append(decl)
        return len(self.program.buffers) - 1

    # -- code --------------------------------------------------------------
    @property
    def pc(self) -> int:
        return len(self.code)

    def emit(self, variant: str, /, site: int | None = None, **fields: Any) -> int:
        self.code.append(Instr(variant, **fields))
        self.code_sites.append(SITE_NONE if site is None else site)
        return len(self.code) - 1

    def emit_cast(
        self,
        dst: Reg,
        src: Operand,
        from_ty: Ty,
        to_ty: Ty,
        rnd: str = "Default",
        sat: bool = False,
    ) -> int:
        """``Cast`` (its ``from`` field is a Python keyword)."""
        fields: dict[str, Any] = {"from": from_ty, "to": to_ty}
        return self.emit("Cast", dst=dst, src=src, rnd=rnd, sat=sat, **fields)

    def patch(self, pc: int, variant: str, /, **fields: Any) -> None:
        self.code[pc] = Instr(variant, **fields)

    def redirect(self, code: list[Instr], sites: list[int]) -> tuple[list[Instr], list[int]]:
        """Emit into ``code``/``sites`` until restored; returns the previous targets."""
        previous = (self.code, self.code_sites)
        self.code, self.code_sites = code, sites
        return previous

    def finish(self) -> Program:
        self.code, self.code_sites = self.program.code, self.program.code_sites
        if not self.code or self.code[-1].variant != "Exit":
            self.emit("Exit")
        return self.program


__all__ = [
    "FORMAT_VERSION",
    "SCHEMA",
    "SITE_NONE",
    "BufferDecl",
    "Const",
    "DimExpr",
    "Instr",
    "Launch",
    "Module",
    "OpKey",
    "Operand",
    "ParamSlot",
    "PredProgram",
    "Program",
    "ProgramBuilder",
    "Reg",
    "RegDecl",
    "Requirements",
    "SiteInfo",
    "SourceSpan",
    "TensorMapSpec",
    "Ty",
    "UnrepresentableType",
    "bulk_completion",
    "mem_mods",
    "opnd",
    "phase_parity",
    "phase_state",
]
