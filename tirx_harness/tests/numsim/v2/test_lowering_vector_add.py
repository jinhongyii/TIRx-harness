"""Lowering of small TIRx kernels to the provisional v2 ``Program``.

Assertions are on Program content (instructions, tables, sites), never on a
rendered text form.
"""

from __future__ import annotations

import json
import struct

import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim.v2.lowering import LoweringUnsupported, lower
from tirx_harness.numsim.v2.lowering import program_builder as pb


VECTOR_ADD = '''
@T.prim_func
def vadd(a: T.Buffer((1024,), "float32"), b: T.Buffer((1024,), "float32"),
         c: T.Buffer((1024,), "float32")):
    T.device_entry()
    bx = T.cta_id([8])
    tx = T.thread_id([128])
    i = bx * 128 + tx
    if i < 1024:
        c[i] = a[i] + b[i]
'''


def _kernel(source: str):
    return tvm.script.from_source(source, {"T": T})


def _ops(program: pb.Program) -> list[type]:
    return [type(instr) for instr in program.code]


def _const_value(program: pb.Program, operand: pb.Operand) -> int:
    assert isinstance(operand, pb.Const)
    return program.consts[operand.index].bits


def test_vector_add_tables():
    program = lower(_kernel(VECTOR_ADD))

    assert program.name == "vadd"
    assert program.unsupported == []
    assert program.topology == pb.Launch(clusters=8, ctas_per_cluster=1, threads_per_cta=128)
    assert program.topology.warps_per_cta == 4

    assert [(s.name, s.kind, s.param_index, s.dtype, s.shape) for s in program.host_abi] == [
        ("a", "buffer", 0, "float32", (1024,)),
        ("b", "buffer", 1, "float32", (1024,)),
        ("c", "buffer", 2, "float32", (1024,)),
    ]
    assert [(b.name, b.space, b.param_slot, b.byte_size) for b in program.buffers] == [
        ("a", pb.Space.GLOBAL, 0, 4096),
        ("b", pb.Space.GLOBAL, 1, 4096),
        ("c", pb.Space.GLOBAL, 2, 4096),
    ]


def test_vector_add_code():
    program = lower(_kernel(VECTOR_ADD))
    code = program.code

    assert _ops(program) == [
        pb.ReadSpecial, pb.ReadSpecial,          # bx, tx
        pb.Binary, pb.Binary, pb.Mov,            # i = bx * 128 + tx
        pb.Compare, pb.If,                       # if i < 1024
        pb.Load, pb.Load, pb.Binary, pb.Store,   # c[i] = a[i] + b[i]
        pb.EndIf, pb.Exit,
    ]
    bx, tx = code[0], code[1]
    assert bx.special is pb.SpecialReg.CTA_LINEAR
    assert tx.special is pb.SpecialReg.THREAD_IN_CTA
    assert program.regs[bx.dst.index].uniform
    assert not program.regs[tx.dst.index].uniform

    mul, add, mov = code[2], code[3], code[4]
    assert (mul.op, mul.dtype, mul.a) == ("Mul", "int32", bx.dst)
    assert _const_value(program, mul.b) == 128
    assert (add.op, add.a, add.b) == ("Add", mul.dst, tx.dst)
    i = mov.dst
    assert mov.src == add.dst
    assert program.regs[i.index] == pb.RegDecl(dtype="int32", name="i", uniform=False)

    compare, branch = code[5], code[6]
    assert (compare.op, compare.dtype, compare.a) == ("Lt", "int32", i)
    assert _const_value(program, compare.b) == 1024
    assert program.regs[compare.dst.index].dtype == "bool"
    assert branch.cond == compare.dst
    assert branch.else_pc == branch.end_pc == 11
    assert isinstance(code[branch.end_pc], pb.EndIf)

    load_a, load_b, fadd, store = code[7:11]
    assert (load_a.buf, load_a.offset, load_a.space, load_a.dtype) == (0, i, pb.Space.GLOBAL, "float32")
    assert (load_b.buf, load_b.offset) == (1, i)
    assert (fadd.op, fadd.dtype, fadd.a, fadd.b) == ("Add", "float32", load_a.dst, load_b.dst)
    assert (store.buf, store.offset, store.value, store.dtype) == (2, i, fadd.dst, "float32")


def test_vector_add_sites_point_at_source():
    program = lower(_kernel(VECTOR_ADD))
    loads = [instr for instr in program.code if isinstance(instr, pb.Load)]
    store = next(instr for instr in program.code if isinstance(instr, pb.Store))

    for instr, buf in [(loads[0], 0), (loads[1], 1), (store, 2)]:
        site = program.sites[instr.site]
        assert site.buffer == buf
        assert len(site.spans) == 1
        # Line 10 of VECTOR_ADD is ``c[i] = a[i] + b[i]``.
        assert site.spans[0].line == 10
    assert program.sites[store.site].kind == "tirx.BufferStore"
    assert program.sites[loads[0].site].kind == "ir.TensorLoad"
    assert program.sites[loads[0].site].spans[0].column < program.sites[loads[1].site].spans[0].column


def test_program_json_is_serde_shaped():
    program = lower(_kernel(VECTOR_ADD))
    decoded = json.loads(program.to_json())
    assert decoded == program.to_dict()

    assert decoded["code"][-1] == "Exit"
    assert decoded["code"][-2] == "EndIf"
    load = decoded["code"][7]
    assert set(load) == {"Load"}
    assert load["Load"]["space"] == "Global"
    assert load["Load"]["offset"] == {"Reg": program.code[7].offset.index}
    assert decoded["topology"]["threads_per_cta"] == 128
    assert decoded["host_abi"][2]["name"] == "c"
    assert {"dtype": "int32", "bits": 1024} in decoded["consts"]


def test_serial_loop_lowers_to_loop_frame():
    program = lower(_kernel('''
@T.prim_func
def scale(x: T.Buffer((32, 4), "float32")):
    T.device_entry()
    T.cta_id([1])
    lane = T.thread_id([32])
    for j in range(4):
        x[lane, j] = x[lane, j] * T.float32(2.0)
'''))
    ops = _ops(program)
    begin = ops.index(pb.LoopBegin)
    loop_begin = program.code[begin]
    head_pc = begin + 1
    end = program.code[loop_begin.end_pc]
    assert isinstance(end, pb.LoopEnd) and end.head_pc == head_pc
    compare = program.code[head_pc]
    loop_if = program.code[head_pc + 1]
    assert isinstance(compare, pb.Compare) and compare.op == "Lt"
    assert isinstance(loop_if, pb.LoopIf) and loop_if.cond == compare.dst
    assert loop_if.end_pc == loop_begin.end_pc
    # The loop bound is uniform, so the counter and its predicate are uniform.
    assert program.regs[compare.a.index].uniform

    body = program.code[head_pc + 2:loop_begin.end_pc]
    load = next(instr for instr in body if isinstance(instr, pb.Load))
    store = next(instr for instr in body if isinstance(instr, pb.Store))
    assert load.buf == store.buf == 0
    # No CSE in the skeleton: each access computes ``lane * 4 + j`` itself.
    counter = compare.a
    for access in (load, store):
        offset_def = next(
            instr for instr in body
            if isinstance(instr, pb.Binary) and instr.dst == access.offset
        )
        assert offset_def.op == "Add" and offset_def.b == counter
    two = next(instr for instr in body if isinstance(instr, pb.Binary) and instr.dtype == "float32")
    assert program.consts[two.b.index] == pb.Scalar(
        "float32", struct.unpack("<I", struct.pack("<f", 2.0))[0]
    )
    # Row-major addressing: lane * 4 + j.
    stride = next(
        instr for instr in program.code[:begin] + body
        if isinstance(instr, pb.Binary) and instr.op == "Mul" and instr.dtype == "int32"
    )
    assert program.consts[stride.b.index].bits == 4


def test_unsupported_builtin_fails_closed():
    source = '''
@T.prim_func
def sync(x: T.Buffer((32,), "int32")):
    T.device_entry()
    lane = T.thread_id([32])
    T.cuda.cta_sync()
    x[lane] = 1
'''
    with pytest.raises(LoweringUnsupported) as raised:
        lower(_kernel(source))
    assert any("tirx.cuda.cta_sync" in reason for reason in raised.value.reasons)

    program = lower(_kernel(source), strict=False)
    gaps = [instr for instr in program.code if isinstance(instr, pb.Unsupported)]
    assert len(gaps) == 1
    assert program.sites[gaps[0].site].kind == "ir.Call"
    # Supported code around the gap is still lowered.
    assert any(isinstance(instr, pb.Store) for instr in program.code)
