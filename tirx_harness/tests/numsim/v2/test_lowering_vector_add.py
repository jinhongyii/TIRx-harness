"""Vector add and basic control flow lowered to ``numsim_core::Program``.

Assertions are on Program content (instructions, tables, sites), never on a
rendered text form.
"""

from __future__ import annotations

import json
import struct

import pytest

from tirx_harness.numsim.v2.lowering import LoweringUnsupported, lower_module
from tirx_harness.numsim.v2.lowering import program_builder as pb

from ._program import const, definition, only, variants

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

F32 = pb.Ty("F32")
S32 = pb.Ty("S32")


def test_vector_add_tables(lower_source):
    program = lower_source(VECTOR_ADD)
    assert program.name == "vadd"
    assert program.unsupported == []
    topo = program.topology
    assert (topo.grid[0], topo.cluster, topo.block) == (pb.DimExpr.const(8), (1, 1, 1), (128, 1, 1))
    assert [(s.name, s.kind, s.param_index, s.dtype, s.shape, s.buf) for s in program.host_abi] == [
        ("a", "Buffer", 0, F32, (pb.DimExpr.const(1024),), 0),
        ("b", "Buffer", 1, F32, (pb.DimExpr.const(1024),), 1),
        ("c", "Buffer", 2, F32, (pb.DimExpr.const(1024),), 2),
    ]
    assert [(b.name, b.space, b.param_slot, b.view_of) for b in program.buffers] == [
        ("a", "Global", 0, None), ("b", "Global", 1, None), ("c", "Global", 2, None),
    ]


def test_vector_add_code(lower_source):
    program = lower_source(VECTOR_ADD)
    code = program.code
    assert variants(program) == [
        "ReadSpecial", "ReadSpecial",            # bx, tx
        "Binary", "Binary", "Mov",               # i = bx * 128 + tx
        "Compare", "If",                         # if i < 1024
        "Load", "Load", "Binary", "Store",       # c[i] = a[i] + b[i]
        "EndIf", "Exit",
    ]
    bx, tx = code[0], code[1]
    assert (bx.sreg, tx.sreg) == ("CtaLinear", "ThreadInCta")
    assert program.regs[bx.dst.index].uniform and not program.regs[tx.dst.index].uniform

    mul, add, mov = code[2:5]
    assert (mul.op, mul.ty, mul.a, const(program, mul.b)) == ("Mul", S32, bx.dst, 128)
    assert (add.op, add.a, add.b, mov.src) == ("Add", mul.dst, tx.dst, add.dst)
    i = mov.dst
    assert program.regs[i.index] == pb.RegDecl(ty=S32, name="i", uniform=False)

    compare, branch = code[5], code[6]
    assert (compare.op, compare.ty, compare.a, const(program, compare.b)) == ("Lt", S32, i, 1024)
    assert program.regs[compare.dst.index].ty == pb.Ty("Pred")
    assert (branch.cond, branch.else_pc, branch.end_pc, branch.elect) == (compare.dst, 11, 11, False)

    load_a, load_b, fadd, store = code[7:11]
    assert (load_a.buf, load_a.offset, load_a.ty, load_a.sem) == (0, i, F32, "Weak")
    assert (load_b.buf, load_b.offset) == (1, i)
    assert (fadd.op, fadd.ty, fadd.a, fadd.b) == ("Add", F32, load_a.dst, load_b.dst)
    assert (store.buf, store.offset, store.value) == (2, i, fadd.dst)


def test_vector_add_sites_point_at_source(lower_source):
    program = lower_source(VECTOR_ADD)
    sites = {pc: program.site_of(pc) for pc, instr in enumerate(program.code)
             if instr.variant in ("Load", "Store")}
    assert len(sites) == 3
    for pc, site in sites.items():
        assert site.spans[0].line == 10            # ``c[i] = a[i] + b[i]``
        assert site.buffer == ["a", "b", "c"][program.code[pc].buf]
    # Pure ALU instructions carry no site (SiteId::NONE).
    assert program.code_sites[2] == pb.SITE_NONE


def test_module_json_is_serde_shaped(lower_source):
    import tvm
    from tvm.script import tirx as T

    module = lower_module(tvm.script.from_source(VECTOR_ADD, {"T": T}))
    decoded = json.loads(module.to_json())
    assert decoded["format_version"] == pb.FORMAT_VERSION
    kernel = decoded["kernels"][0]
    assert kernel["code"][-1] == "Exit" and kernel["code"][-2] == "EndIf"
    assert len(kernel["code_sites"]) == len(kernel["code"])
    load = kernel["code"][7]["Load"]
    assert load["ty"] == {"elem": "F32", "lanes": 1}
    assert load["offset"] == {"Reg": module.kernels[0].code[7].offset.index}
    assert kernel["code"][7 + 3]["Store"]["buf"] == 2
    assert kernel["buffers"][0]["space"] == "global"
    assert kernel["topology"]["grid"][0] == {"Const": 8}
    assert {"ty": {"elem": "S32", "lanes": 1}, "bits": 1024} in kernel["consts"]


def test_serial_loop_lowers_to_loop_frame(lower_source):
    program = lower_source('''
@T.prim_func
def scale(x: T.Buffer((32, 4), "float32")):
    T.device_entry()
    T.cta_id([1])
    lane = T.thread_id([32])
    for j in range(4):
        x[lane, j] = x[lane, j] * T.float32(2.0)
''')
    begin = only(program, "LoopBegin")
    begin_pc = program.code.index(begin)
    end = program.code[begin.end_pc]
    assert end.variant == "LoopEnd" and end.head_pc == begin_pc + 1
    compare, loop_if = program.code[begin_pc + 1], program.code[begin_pc + 2]
    assert compare.variant == "Compare" and compare.op == "Lt"
    assert loop_if.variant == "LoopIf" and loop_if.cond == compare.dst and loop_if.end_pc == begin.end_pc
    assert program.regs[compare.a.index].uniform      # uniform bounds => uniform counter
    assert program.site_of(begin_pc).kind == "tirx.For"  # LoopBegin's site identifies the loop
    two = next(i for i in program.code if i.variant == "Binary" and i.ty == F32)
    assert program.consts[two.b.index] == (F32, struct.unpack("<I", struct.pack("<f", 2.0))[0])
    store = only(program, "Store")
    offset = definition(program, store.offset)
    assert offset.variant == "Binary" and offset.op == "Add" and offset.b == compare.a  # lane*4 + j


def test_unsupported_fails_closed(lower_source):
    source = '''
@T.prim_func
def bad(x: T.Buffer((32,), "int32")):
    T.device_entry()
    lane = T.thread_id([32])
    x[lane] = T.call_extern("int32", "mystery", lane)
'''
    with pytest.raises(LoweringUnsupported) as raised:
        lower_source(source)
    assert any("mystery" in r or "call_extern" in r for r in raised.value.reasons)
    program = lower_source(source, strict=False)
    gap = only(program, "Unsupported")
    assert "call_extern" in program.strings[gap.reason] or "builtin" in program.strings[gap.reason]
    assert program.site_of(program.code.index(gap)) is not None


def test_half_chain_stays_f32_until_the_store(lower_source):
    """Ruling D1: ``a*b+c`` in float16 computes in f32 and rounds once, at the store."""
    program = lower_source('''
@T.prim_func
def fma16(a: T.Buffer((32,), "float16"), b: T.Buffer((32,), "float16"),
          c: T.Buffer((32,), "float16"), d: T.Buffer((32,), "float16")):
    T.device_entry()
    lane = T.thread_id([32])
    d[lane] = a[lane] * b[lane] + c[lane]
''')
    f16, f32 = pb.Ty("F16"), F32
    loads = [i for i in program.code if i.variant == "Load"]
    assert [i.ty for i in loads] == [f16, f16, f16]
    widen = [i for i in program.code if i.variant == "Cast" and i.fields["to"] == f32]
    assert [i.src for i in widen] == [ld.dst for ld in loads]   # each leaf widened exactly once
    mul, add = [i for i in program.code if i.variant == "Binary"]
    assert (mul.op, mul.ty, add.op, add.ty) == ("Mul", f32, "Add", f32)
    assert add.a == mul.dst                                      # the product is not rounded to half
    narrow = [i for i in program.code if i.variant == "Cast" and i.fields["to"] == f16]
    assert len(narrow) == 1 and narrow[0].src == add.dst
    store = only(program, "Store")
    assert (store.ty, store.value) == (f16, narrow[0].dst)
