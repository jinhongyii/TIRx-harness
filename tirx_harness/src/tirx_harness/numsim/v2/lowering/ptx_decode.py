"""Pure-Python decoder for the table-driven ``tirx.ptx.*`` Call ABI.

Wire layout (TVM ``tvm.backend.cuda.ptx``)::

    [present operand lanes..., instruction predicate?] [one StringImm per modifier slot] [marker]

The marker is a comma list of ``pred`` (instruction predicate present),
``keep`` (preserve destination under a false predicate), ``pN`` (lane N is a
predicate register) and ``sN`` (lane N is sunk).

This is the same algorithm as the legacy ``transpiler/ptx_dialect.py`` minus
the native ``ptx_call_parts`` helper; it returns per-operand slot metadata as
well so the lowering never has to consult the table again.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from tvm.backend.cuda.ptx.table import (
    TABLE,
    lanes_of,
    mods,
    operand_dtypes,
    operand_layout,
    operand_space,
    operand_type,
)


class PtxDecodeError(ValueError):
    pass


SINK = None
"""A sunk destination lane (``sN`` marker)."""


@dataclass(frozen=True)
class OperandInfo:
    name: str
    kind: str            # reg | addr | ptr | imm
    rw: str              # r | w | rw
    lanes: int
    ptx_type: str        # "" when the table leaves it untyped
    dtype: str           # canonical TVM dtype ("" when untyped)
    space: str           # addr operands only
    literal: str | None


@dataclass(frozen=True)
class DecodedPtx:
    op_name: str
    table_name: str
    modifiers: tuple[tuple[str, str], ...]
    operands: tuple[OperandInfo, ...]
    values: tuple[tuple[Any, ...], ...]   # per operand; literal imm -> (literal,)
    predicate: Any | None
    preserve_dst: bool
    orders_memory: bool

    def modifier(self, name: str) -> str:
        return dict(self.modifiers).get(name, "")

    @property
    def mod_tokens(self) -> tuple[str, ...]:
        """Self-describing ``OpKey.mods``: ``slot=token`` for every non-empty slot."""
        return tuple(f"{slot}={token}" for slot, token in self.modifiers if token)


@cache
def _by_op_name() -> dict[str, Any]:
    return {entry.op_name: entry for entry in TABLE.values()}


def is_table_op(op_name: str) -> bool:
    return op_name in _by_op_name()


def _string_value(node: Any) -> str | None:
    info = getattr(type(node), "__tvm_ffi_type_info__", None)
    if info is not None and info.type_key == "ir.StringImm":
        return str(node.value)
    return None


def _marker(op_name: str, marker: str) -> tuple[bool, bool, frozenset[int], frozenset[int]]:
    flags = marker.split(",") if marker else []
    predicated = preserve = False
    preds: set[int] = set()
    sinks: set[int] = set()
    for flag in flags:
        if flag == "pred":
            predicated = True
        elif flag == "keep":
            preserve = True
        elif len(flag) >= 2 and flag[0] in "ps" and flag[1:].isdigit():
            (preds if flag[0] == "p" else sinks).add(int(flag[1:]))
        else:
            raise PtxDecodeError(f"{op_name}: unknown marker flag {flag!r}")
    if preserve and not predicated:
        raise PtxDecodeError(f"{op_name}: 'keep' requires an instruction predicate")
    return predicated, preserve, frozenset(preds), frozenset(sinks)


def decode(call: Any) -> DecodedPtx:
    op_name = str(call.op.name)
    entry = _by_op_name().get(op_name)
    if entry is None:
        raise PtxDecodeError(f"{op_name}: not in the target TVM PTX table")
    args = tuple(call.args)
    meta = len(entry.slots) + 1
    if len(args) < meta:
        raise PtxDecodeError(f"{op_name}: expected {meta} trailing modifier/marker strings")
    tokens = []
    for slot, node in zip(entry.slots, args[-meta:-1]):
        token = _string_value(node)
        if token is None:
            raise PtxDecodeError(f"{op_name}: modifier slot {slot.name!r} is not a StringImm")
        if token and token not in slot.choices:
            raise PtxDecodeError(f"{op_name}: modifier {slot.name}={token!r} not in {slot.choices}")
        if not token and not slot.optional:
            raise PtxDecodeError(f"{op_name}: required modifier slot {slot.name!r} is empty")
        tokens.append(token)
    marker = _string_value(args[-1])
    if marker is None:
        raise PtxDecodeError(f"{op_name}: trailing marker is not a StringImm")
    mod_map = mods(entry, tuple(tokens))
    if entry.check is not None:
        error = entry.check(mod_map)
        if error:
            raise PtxDecodeError(f"{op_name}: illegal modifier combination: {error}")
    predicated, preserve, _preds, sinks = _marker(op_name, marker)

    layout = operand_layout(entry, mod_map)
    lane_total = sum(lanes for _, _, lanes in layout)
    serialized = args[:-meta]
    expected = lane_total - len(sinks) + int(predicated)
    if len(serialized) != expected:
        raise PtxDecodeError(f"{op_name}: expected {expected} operand args, got {len(serialized)}")
    predicate = serialized[-1] if predicated else None
    present = iter(serialized[:-1] if predicated else serialized)

    infos: list[OperandInfo] = []
    values: list[tuple[Any, ...]] = []
    rows = {id(slot): (first, lanes) for slot, first, lanes in layout}
    for slot in entry.operands:
        ptx_type = dtype = ""
        if slot.kind == "reg":
            try:
                ptx_type = operand_type(slot, mod_map)
                dtype = operand_dtypes(slot, mod_map)[0]
            except KeyError:
                pass
        space = operand_space(slot, mod_map) if slot.kind == "addr" else ""
        if slot.kind == "imm" and slot.literal is not None:
            infos.append(OperandInfo(slot.name, "imm", slot.rw, 1, ptx_type, dtype, space, str(slot.literal)))
            values.append((slot.literal,))
            continue
        first, lanes = rows[id(slot)]
        infos.append(OperandInfo(slot.name, slot.kind, slot.rw, lanes_of(slot, mod_map),
                                 ptx_type, dtype, space, None))
        values.append(tuple(SINK if first + lane in sinks else next(present) for lane in range(lanes)))
    return DecodedPtx(
        op_name=op_name,
        table_name=entry.name,
        modifiers=tuple(mod_map.items()),
        operands=tuple(infos),
        values=tuple(values),
        predicate=predicate,
        preserve_dst=preserve,
        orders_memory=bool(entry.orders_memory),
    )


__all__ = ["DecodedPtx", "OperandInfo", "PtxDecodeError", "SINK", "decode", "is_table_op"]
