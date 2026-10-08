"""Lowering (Program content): the ``.pred`` carrier bridge of table PTX ops, and
the size of a committed ``shared.dyn`` pool (CONTRACT_REQUESTS "W11-other-assertion"
W11-1 and W11-3)."""

from __future__ import annotations

from tirx_harness.numsim.v2.lowering import program_builder as pb

from ._program import all_of, definition, op_key, pc_of

GUARDED_PRED = '''
@T.prim_func
def k(out: T.Buffer((2, 32), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    d = T.alloc_local((2,), "uint32")
    d[0] = T.uint32(91)
    d[1] = T.uint32(91)
    T.ptx["setp.lt.s32"](d[0], T.int32(1), T.int32(2), pred=lane < 4, preserve_dst=True)
    T.ptx["setp.lt.s32"](d[1], T.int32(1), T.int32(2), pred=lane < 4)
    out[0, lane] = d[0]
    out[1, lane] = d[1]
'''


def _guarded_ptx(program: pb.Program, name: str) -> list[tuple[int, pb.Instr]]:
    return [(pc_of(program, i), i) for i in all_of(program, "Ptx") if op_key(program, i).name == name]


def test_guarded_pred_destination_is_bridged_before_the_guard(lower_source):
    program = lower_source(GUARDED_PRED)
    (keep_pc, keep), (undef_pc, undef) = _guarded_ptx(program, "tirx.ptx.setp")
    for pc in (keep_pc, undef_pc):
        assert program.code[pc - 1].variant == "If"  # the op itself stays inside the guard
    # preserve_dst: just before the If, dst := (dst != 0) as 0/1 (TVM `_pred_keep`:
    # setp.ne.b32 pd, %0, 0 ... selp.b32 %0, 1, 0, pd on every lane).
    (dst,) = keep.dsts
    movs = [i for i in program.code[:keep_pc] if i.variant == "Mov" and i.dst == dst]
    bridge = movs[-1]
    to_u32 = definition(program, bridge.src) if isinstance(bridge.src, pb.Reg) else None
    assert to_u32 is not None and to_u32.variant == "Cast"
    assert getattr(to_u32, "to") == pb.Ty("U32")
    to_pred = definition(program, to_u32.src)
    assert to_pred.variant == "Cast" and getattr(to_pred, "to") == pb.Ty("Pred")
    # Without preserve_dst the off-lane value is 0 (`_pred_undef`: selp of an
    # unwritten predicate), written without reading the carrier.
    (dst,) = undef.dsts
    movs = [i for i in program.code[keep_pc:undef_pc] if i.variant == "Mov" and i.dst == dst]
    assert movs and isinstance(movs[-1].src, pb.Const)
    assert program.consts[movs[-1].src.index][1] == 0


def test_unguarded_pred_sources_are_bridged(lower_source):
    program = lower_source('''
@T.prim_func
def k(a: T.Buffer((32,), "uint32"), out: T.Buffer((32,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    d = T.alloc_local((1,), "uint32")
    T.ptx["and.pred"](d[0], T.ptx.pred(a[lane]), T.ptx.pred(a[lane]))
    out[lane] = d[0]
''')
    (op,) = [i for i in all_of(program, "Ptx") if op_key(program, i).name.endswith(".and")] or all_of(program, "Ptx")
    for src in op.srcs:
        cast = definition(program, src)
        assert cast.variant == "Cast" and getattr(cast, "to") == pb.Ty("U32")
        assert definition(program, cast.src).variant == "Cast"
    # No pre-guard bridge for an unguarded op.
    assert all(i.variant != "If" for i in program.code)


def test_committed_dyn_smem_bounds_the_pool_not_its_strided_view(lower_source):
    program = lower_source('''
@T.prim_func
def k():
    T.attr({"tirx.device_entry": T.bool(True), "tirx.dyn_smem_bytes": 2048})
    lane = T.lane_id([32])
    T.warp_id([1])
    pool = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    scratch = T.decl_buffer((2, 4, 64), "float32", data=pool.data, strides=(8192, 64, 1), scope="shared.dyn", align=16)
    if lane == 0:
        scratch[1, 0, 0] = T.float32(1)
''')
    names = [b.name for b in program.buffers]
    pool = program.buffers[names.index("pool")]
    scratch = program.buffers[names.index("scratch")]
    assert pool.byte_len == pb.DimExpr.const(2048)
    # The view keeps its strided extent; the engine bounds-checks it against the pool.
    assert scratch.view_of == names.index("pool") and scratch.byte_len == pb.DimExpr.const(33792)
