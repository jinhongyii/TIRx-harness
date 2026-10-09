"""Small helpers for asserting on lowered Programs (content, never text)."""

from __future__ import annotations

from tirx_harness.numsim.v2.lowering import program_builder as pb


def variants(program: pb.Program) -> list[str]:
    return [instr.variant for instr in program.code]


def only(program: pb.Program, variant: str) -> pb.Instr:
    found = [instr for instr in program.code if instr.variant == variant]
    assert len(found) == 1, f"expected one {variant}, got {len(found)}: {variants(program)}"
    return found[0]


def all_of(program: pb.Program, variant: str) -> list[pb.Instr]:
    return [instr for instr in program.code if instr.variant == variant]


def pc_of(program: pb.Program, instr: pb.Instr) -> int:
    return next(i for i, x in enumerate(program.code) if x is instr)


def definition(program: pb.Program, reg: pb.Reg) -> pb.Instr:
    """The last instruction before use that writes ``reg``."""
    writers = [instr for instr in program.code if reg in instr.writes()]
    assert writers, f"{reg} has no writer"
    return writers[-1]


def const(program: pb.Program, operand: pb.Operand) -> int:
    assert isinstance(operand, pb.Const), operand
    return program.consts[operand.index][1]


def op_key(program: pb.Program, instr: pb.Instr) -> pb.OpKey:
    assert instr.variant == "Ptx"
    return program.ops[instr.op]
