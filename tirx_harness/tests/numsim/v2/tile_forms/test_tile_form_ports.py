"""Program content of the v2 tile-form ports (copy / fill / permute_layout / reduce).

These forms are the ones TVM's ``TilePrimitiveDispatch`` does not register
(amended Decision 6); ``v2/lowering/tile_forms`` ports legacy's canonical
semantics onto ordinary Program ops. Assertions are on instructions, never
on text.
"""

from __future__ import annotations

from tirx_harness.numsim.v2.lowering import program_builder as pb
from tirx_harness.numsim.v2.lowering.tile_forms import copy as tile_copy
from tirx_harness.numsim.v2.lowering.tile_forms import fill as tile_fill
from tirx_harness.numsim.v2.lowering.tile_forms import reduce as tile_reduce

from .._program import all_of, const, pc_of

FULL = 0xFFFFFFFF

PERMUTE = '''
@T.prim_func
def k(source: T.Buffer((128,), "uint32"), output: T.Buffer((128,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    src = T.alloc_shared((128,), "uint32")
    dst = T.alloc_shared((128,), "uint32", layout=T.TileLayout(T.S[(4, 32):(1, 4)]))
    src[lane] = source[lane]
    T.cuda.warp_sync()
    T.warp.permute_layout(dst[0:128], src[0:128])
    output[lane] = dst[lane]
'''


def _buf(program: pb.Program, name: str) -> int:
    return [b.name for b in program.buffers].index(name)


def _full_warp_vote(program: pb.Program) -> pb.Instr:
    votes = [v for v in all_of(program, "Vote") if v.mode == "All"]
    assert len(votes) == 1, [i.variant for i in program.code]
    assert const(program, votes[0].membermask) == FULL
    return votes[0]


def test_permute_layout_is_a_warp_snapshot_copy(lower_source):
    program = lower_source(PERMUTE)
    src, dst = _buf(program, "src"), _buf(program, "dst")
    vote = _full_warp_vote(program)
    syncs = [i for i in all_of(program, "WarpSync") if const(program, i.membermask) == FULL]
    # The kernel's own warp_sync, then the snapshot sync and the completion sync.
    assert len(syncs) == 3
    snapshot, completion = syncs[1], syncs[2]
    reads = [i for i in all_of(program, "Load") if i.buf == src]
    writes = [i for i in all_of(program, "Store") if i.buf == dst]
    assert len(reads) == 1 and len(writes) == 1
    # participation -> every owner reads -> warp sync -> owners write -> warp sync
    assert pc_of(program, vote) < pc_of(program, reads[0]) < pc_of(program, snapshot)
    assert pc_of(program, snapshot) < pc_of(program, writes[0]) < pc_of(program, completion)
    # The snapshot lives in per-thread registers (indexed by the element), not memory.
    assert all_of(program, "StoreRegIndexed") and all_of(program, "LoadRegIndexed")
    # Element i is read and written by lane i % 32 of the warp (legacy scope_lanes):
    # a loop over 128 / 32 slots.
    assert len(all_of(program, "LoopBegin")) == 2


def test_shared_cta_reduction_is_sequential_with_identity_and_completion_barrier(lower_source):
    program = lower_source('''
@T.prim_func
def k(source: T.Buffer((64, 4), "float16"), output: T.Buffer((64,), "float16")):
    T.attr({"tirx.device_entry": T.bool(True)})
    _cta = T.cta_id([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread: T.int32 = warp * 32 + lane
    src = T.alloc_shared((64, 4), "float16")
    res = T.alloc_shared((64,), "float16")
    for column in range(4):
        src[thread, column] = source[thread, column]
    T.cuda.cta_sync()
    T.cta.max(res[0:64], src[0:64, 0:4], [1], False, dispatch="shared")
    output[thread] = res[thread]
''')
    src, res = _buf(program, "src"), _buf(program, "res")
    # No warp participation vote at CTA scope; legacy had no CTA rendezvous here either.
    assert not [v for v in all_of(program, "Vote") if v.mode == "All"]
    store = next(i for i in all_of(program, "Store") if i.buf == res)
    barriers = all_of(program, "Barrier")
    # The kernel's cta_sync, then the reduction's completion bar.sync 0 (all threads).
    assert len(barriers) == 2 and barriers[1].kind == "Sync" and const(program, barriers[1].id) == 0
    assert barriers[1].count is None and pc_of(program, store) < pc_of(program, barriers[1])
    read = next(i for i in all_of(program, "Load") if i.buf == src)
    combine = [i for i in all_of(program, "Binary") if i.op == "Max"]
    assert combine and pc_of(program, read) < pc_of(program, combine[0])
    # The accumulator starts from max's identity: -65504 (f16 lowest finite).
    seeds = [i for i in all_of(program, "Mov") if isinstance(i.src, pb.Const)
             and program.consts[i.src.index][0] == pb.Ty("F16")]
    assert [program.consts[s.src.index][1] for s in seeds] == [0xFBFF]


def test_partial_lane_fill_requires_the_full_warp(lower_source):
    program = lower_source('''
@T.prim_func
def k():
    T.attr({"tirx.device_entry": T.bool(True)})
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tile = T.alloc_local((128, 1), layout=T.TileLayout(T.S[(128, 1):(1 @ Axis.tid_in_wg, 1)]))
    if warp == 0 and lane < 16:
        T.wg.fill(tile[0:128, 0], T.float32(1.0))
''')
    vote = _full_warp_vote(program)
    # The vote sits under the kernel's `if`: lanes 16..31 are not active there, so
    # the engine reports warp_collective_divergence (legacy #594 participation).
    first_if = all_of(program, "If")[0]
    assert pc_of(program, first_if) < pc_of(program, vote)
    # A local fill completes without any barrier (no warpgroup rendezvous).
    assert not all_of(program, "Barrier")
    # Each thread stores only the elements its tid_in_wg owns: a guarded loop.
    assert all_of(program, "LoopBegin") and all_of(program, "Store")


def test_repairs_drop_dispatch_hints_and_type_fill_literals(lower_source):
    import tvm
    from tvm.script import tirx as T

    func = tvm.script.from_source('''
@T.prim_func
def k(output: T.Buffer((32,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    mask = T.alloc_local((1,), "uint32")
    T.tile.fill(mask[0:1], 4294967295, dispatch="reg")
    T.tile.copy(output[lane:lane + 1], mask[0:1], dispatch="gmem_smem")
    T.tile.sum(mask[0:1], output[0:32], [0], False, dispatch="shared")
''', {"T": T})
    calls = []
    from tvm import tirx
    from tvm_ffi import structural_visit

    structural_visit(func.body, [(tirx.TilePrimitiveCall, lambda n, v: calls.append(n))])
    fill, copy, total = calls
    fixed = tile_fill.repair(fill)
    assert fixed.dispatch is None
    value = fixed.args[1]
    assert (str(value.ty.dtype), int(value.value)) == ("uint32", FULL)
    for call, repair in ((copy, tile_copy.repair), (total, tile_reduce.repair)):
        fixed = repair(call)
        assert fixed.dispatch is None and len(fixed.args) == len(call.args)
        assert repair(fixed) is None  # nothing left to repair


def test_snapshot_copy_with_runtime_minimum_evaluates_it_once(lower_source):
    program = lower_source('''
@T.prim_func
def k(source: T.Buffer((2, 32), "float32"), output: T.Buffer((32,), "float32"), row: T.Buffer((1,), "int32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    src = T.alloc_shared((2, 32))
    dst = T.alloc_shared((2, 32), layout=T.TileLayout(T.S[(2, 32):(1, 2)]))
    src[0, lane] = source[0, lane]
    src[1, lane] = source[1, lane]
    T.cuda.warp_sync()
    T.warp.permute_layout(dst[row[0], 0:32], src[row[0], 0:32])
    output[lane] = dst[0, lane]
''')
    row = _buf(program, "row")
    loads = [i for i in all_of(program, "Load") if i.buf == row]
    # Two region minima (dst, src), each read once before the element loops.
    assert len(loads) == 2
    first_loop = all_of(program, "LoopBegin")[0]
    assert all(pc_of(program, i) < pc_of(program, first_loop) for i in loads)




def test_register_owner_checks_of_a_tile_op_point_at_the_tile_call(lower_source):
    """W11-5: every register-ownership check has a source site; the checks the
    tile op's own loads and stores need point at the tile call (its marker),
    not at TVM's or the tile form's generated statements, which have no
    user span."""
    program = lower_source('''
@T.prim_func
def k(source: T.Buffer((128, 2), "float32"), output: T.Buffer((128, 2), "float32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    src = T.alloc_local((128, 2), layout=T.TileLayout(T.S[(128, 2):(1 @ Axis.tid_in_wg, 1)]))
    dst = T.alloc_local((128, 2), layout=T.TileLayout(T.S[(128, 2):(1 @ Axis.tid_in_wg, 1)]))
    for column in range(2):
        src[thread, column] = source[thread, column]
    T.wg.copy(dst[0:128, 0:2], src[0:128, 0:2])
    for column in range(2):
        output[thread, column] = dst[thread, column]
''')
    checks = [pc for pc, i in enumerate(program.code)
              if i.variant == "Assert" and "owned by another thread" in program.strings[i.msg]]
    # 2 kernel stores + 2 kernel loads, and the copy's snapshot load and store.
    assert len(checks) >= 4
    lines = []
    for pc in checks:
        assert program.code_sites[pc] < len(program.sites), "register-owner check without a site"
        spans = program.sites[program.code_sites[pc]].spans
        assert spans, "register-owner check without a source span"
        lines.append(spans[0].line)
    # The copy is line 12 of the kernel source (1-based, after the leading newline).
    assert lines.count(12) >= 2, lines


LOCAL_WARP_COPY = '''
@T.prim_func
def k(output: T.Buffer((32, 2), "float32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source = T.alloc_local((2,), "float32")
    destination = T.alloc_local((2,), "float32")
    source[0] = T.Cast("float32", lane)
    source[1] = T.Cast("float32", lane)
    T.warp.copy(destination[0:2], source[0:2])
    output[lane, 0] = destination[0]
    output[lane, 1] = destination[1]
'''


def test_tvm_single_thread_fallback_on_registers_takes_the_tile_form(lower_source):
    """F4: TVM picks copy/fallback (one thread) for this register copy; v2 lowers
    it with the tile form, where every lane copies its own registers."""
    import tvm
    from tvm.script import tirx as T
    from tvm_ffi import structural_visit

    from tirx_harness.numsim.v2.lowering import tile_forms
    from tirx_harness.numsim.v2.lowering.ir_walk import dispatch_tile_primitives

    func = tvm.script.from_source(LOCAL_WARP_COPY, {"T": T})
    placeholders = []

    def on_call(node, visitor):
        if getattr(node.op, "name", "") == "tirx.call_extern" and node.args[0].value == tile_forms.PLACEHOLDER:
            placeholders.append(node)

    structural_visit(dispatch_tile_primitives(func).body, [(tvm.ir.Call, on_call)])
    assert len(placeholders) == 1
    program = lower_source(LOCAL_WARP_COPY)
    _full_warp_vote(program)  # the tile form's warp participation
    # Thread-private operands: every lane runs the copy loop, no owner guard.
    assert not all_of(program, "If")


def test_cross_owner_fragment_copy_is_transported_through_shared_scratch(lower_source):
    program = lower_source('''
@T.prim_func
def k(source: T.Buffer((128, 2), "float32"), output: T.Buffer((128, 2), "float32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    src = T.alloc_local((128, 2), layout=T.TileLayout(T.S[(128, 2):(1 @ Axis.tid_in_wg, 1)]))
    dst = T.alloc_local((128, 2), layout=T.TileLayout(T.S[(128, 2):(1 @ Axis.tid_in_wg, 1)]))
    for column in range(2):
        src[thread, column] = source[thread, column]
    T.wg.copy(dst[64:128, 0:2], src[0:64, 0:2])
    for column in range(2):
        output[thread, column] = dst[thread, column]
''')
    scratch = [i for i, b in enumerate(program.buffers) if b.name.endswith(".transport")]
    assert len(scratch) == 1 and program.buffers[scratch[0]].space == "Shared"
    stores = [i for i in all_of(program, "Store") if i.buf == scratch[0]]
    loads = [i for i in all_of(program, "Load") if i.buf == scratch[0]]
    assert len(stores) == 1 and len(loads) == 1
    barriers = [b for b in all_of(program, "Barrier") if const(program, b.id) == 8
                and const(program, b.count) == 128]
    # scope sync -> stage -> scope sync -> restore -> scope sync (legacy transport).
    assert len(barriers) == 3
    pcs = [pc_of(program, b) for b in barriers]
    assert pcs[0] < pc_of(program, stores[0]) < pcs[1] < pc_of(program, loads[0]) < pcs[2]
