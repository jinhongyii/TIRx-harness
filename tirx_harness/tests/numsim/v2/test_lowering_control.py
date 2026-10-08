"""Control flow and waits: While/Break, guarded if_then_else, wait_until predicate programs."""

from __future__ import annotations

from tirx_harness.numsim.v2.lowering import program_builder as pb

from ._program import all_of, const, only


def test_wait_until_predicate_program(lower_source):
    program = lower_source('''
@T.prim_func
def wait(state: T.Buffer((32,), "int32"), out: T.Buffer((32,), "int32")):
    T.device_entry()
    lane = T.thread_id([32])
    seen = T.alloc_local((1,), "int32")
    target: T.int32 = lane + 1
    seen[0] = 999
    T.cuda.wait_until(seen[0], state.ptr_to([lane]), lambda current: current >= target)
    out[lane] = seen[0]
''')
    wait = only(program, "WaitUntil")
    assert (wait.ty, wait.space, wait.sem, wait.scope) == (pb.Ty("S32"), "Global", "Acquire", "Gpu")
    assert program.regs[wait.dst.index].name == "seen" and wait.may_block
    pred = program.preds[wait.pred]
    body = program.code[pred.start:pred.end]
    main_end = pred.start
    assert program.code[main_end - 1].variant == "Exit"            # predicates follow the main body
    assert [i.variant for i in body] == ["Compare"]
    compare = body[0]
    assert (compare.op, compare.a, compare.dst) == ("Ge", pred.arg, pred.result)
    # ``target`` is a captured register of the main program, snapshotted at issue.
    assert wait.captures == [compare.b] and program.regs[compare.b.index].name == "target"
    assert pred.reads_memory is False
    # The polled buffer is marked as holding declared sync words.
    assert program.buffers[program.host_abi[0].buf].sync_words


def test_wait_until_predicate_that_reads_memory(lower_source):
    program = lower_source('''
@T.prim_func
def wait(state: T.Buffer((32,), "int32"), table: T.Buffer((4,), "int32"), out: T.Buffer((32,), "int32")):
    T.device_entry()
    lane = T.thread_id([32])
    seen = T.alloc_local((1,), "int32")
    seen[0] = 0
    T.cuda.wait_until(seen[0], state.ptr_to([lane]), lambda current: table[current] == 1)
    out[lane] = seen[0]
''')
    pred = program.preds[only(program, "WaitUntil").pred]
    body = program.code[pred.start:pred.end]
    assert pred.reads_memory and "Load" in [i.variant for i in body]
    assert all(i.variant in pb.PRED_ALLOWED for i in body)


def test_while_break_and_guarded_select(lower_source):
    program = lower_source('''
@T.prim_func
def k(x: T.Buffer((64,), "int32"), out: T.Buffer((32,), "int32")):
    T.device_entry()
    lane = T.thread_id([32])
    i: T.int32 = 0
    while i < 64:
        if x[i] == lane:
            break
        i = i + 1
    out[lane] = T.if_then_else(i < 64, x[T.min(i, 63)], -1)
''')
    begin = only(program, "LoopBegin")
    loop_if = only(program, "LoopIf")
    brk = only(program, "Break")
    begin_pc, brk_pc = program.code.index(begin), program.code.index(brk)
    assert begin_pc < program.code.index(loop_if) < brk_pc < begin.end_pc
    assert program.code[begin.end_pc].variant == "LoopEnd"
    # if_then_else with a load in an arm executes only the taken arm (If/Else, not Select).
    ifs = all_of(program, "If")
    guarded = ifs[-1]
    else_instr = program.code[guarded.else_pc]
    assert else_instr.variant == "Else" and else_instr.end_pc == guarded.end_pc
    taken = program.code[program.code.index(guarded) + 1:guarded.else_pc]
    assert "Load" in [i.variant for i in taken]
    fallback = program.code[guarded.else_pc + 1:guarded.end_pc]
    assert [i.variant for i in fallback] == ["Mov"] and const(program, fallback[0].src) == (1 << 32) - 1


def test_bitwise_and_logical_not(lower_source):
    program = lower_source('''
@T.prim_func
def k(x: T.Buffer((32,), "uint32"), out: T.Buffer((32,), "uint32")):
    T.device_entry()
    lane = T.thread_id([32])
    if not (lane < 4):
        out[lane] = ~x[lane]
''')
    ops = {(i.op, i.ty) for i in all_of(program, "Unary")}
    assert ops == {("Not", pb.Ty("Pred")), ("BitNot", pb.Ty("U32"))}


def test_module_qualifies_per_launch_slots():
    """V2C-7: a name several kernels declare is kernel-qualified (`k<i>:`); unique names stay bare."""
    import tvm
    from tvm.script import tirx as T

    from tirx_harness.numsim.v2.lowering import lower_module

    def kernel(dtype: str):
        return tvm.script.from_source(f'''
@T.prim_func
def k(a: T.Buffer((32,), "float32"), n: T.{dtype}):
    T.attr({{"tirx.device_entry": T.bool(True)}})
    lane = T.lane_id([32])
    T.warp_id([1])
    a[lane] = a[lane] + T.Cast("float32", n)
''', {"T": T})

    module = lower_module([kernel("int32"), kernel("int32")])
    names = [[(s.name, s.local_name or s.name) for s in p.host_abi] for p in module.kernels]
    assert names[0][0] == ("k0:a", "a") and names[1][0] == ("k1:a", "a")
    assert names[0][1] == ("k0:n", "n") and names[1][1] == ("k1:n", "n")
    single = lower_module([kernel("int32")])
    assert [s.name for s in single.kernels[0].host_abi][:2] == ["a", "n"]


def test_vectorized_loop_lowers_as_serial_loop(lower_source):
    program = lower_source('''
@T.prim_func
def k(out: T.Buffer((128,), "float32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    for v in T.vectorized(4):
        out[lane * 4 + v] = T.float32(1)
''')
    assert not program.unsupported
    assert [i.variant for i in program.code].count("LoopBegin") == 1
