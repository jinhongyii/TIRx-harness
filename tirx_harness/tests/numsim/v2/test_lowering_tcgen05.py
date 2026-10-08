"""tcgen05 family: alloc/dealloc/relinquish, ld/st/wait, descriptor encoders, MMA, commit."""

from __future__ import annotations

from tirx_harness.numsim.v2.lowering import program_builder as pb

from ._program import all_of, const, only, op_key

TCGEN = '''
@T.prim_func
def k(out: T.Buffer((128, 4), "uint32")):
    T.func_attr({"tirx.cuda_arch": "sm_100a"})
    T.attr({"tirx.device_entry": T.bool(True)})
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    tmem_addr = T.alloc_shared((1,), "uint32")
    bar = T.alloc_shared((1,), "uint64")
    a_s = T.alloc_shared((128, 64), "float16", align=1024)
    b_s = T.alloc_shared((128, 64), "float16", align=1024)
    regs = T.alloc_local((4,), "uint32")
    desc_a: T.uint64
    desc_b: T.uint64
    idesc: T.uint32
    if warp == 0:
        T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(tmem_addr[0])), T.uint32(128), "alloc", "cta_group::1", "sync", "aligned", "shared::cta", "b32", "")
    T.cuda.cta_sync()
    taddr: T.uint32 = tmem_addr[0]
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(desc_a), T.address_of(a_s[0, 0]), 16, 64, 3)
        T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(desc_b), T.address_of(b_s[0, 0]), 16, 64, 3)
        T.cuda.tcgen05.encode_instr_descriptor(T.address_of(idesc), d_dtype="float32", a_dtype="float16", b_dtype="float16", M=128, N=128, K=16, trans_a=False, trans_b=False)
        T.ptx.tcgen05(taddr, desc_a, desc_b, idesc, T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.bool(False), "mma", "cta_group::1", "kind::f16", "", "p8")
        T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(bar[0])), "commit", "cta_group::1", "mbarrier::arrive::one", "shared::cluster", "b64", "")
    T.cuda.mbarrier_wait(T.address_of(bar[0]), 0)
    T.ptx.tcgen05(regs[0], regs[1], regs[2], regs[3], taddr, "ld", "sync", "aligned", "32x32b", "x4", "", "b32", "")
    T.ptx.tcgen05("wait::ld", "sync", "aligned", "")
    for i in range(4):
        out[warp * 32 + lane, i] = regs[i]
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.tcgen05(taddr, T.uint32(128), "dealloc", "cta_group::1", "sync", "aligned", "b32", "")
        T.ptx.tcgen05("relinquish_alloc_permit", "cta_group::1", "sync", "aligned", "")
'''


def test_tcgen05_lifecycle_and_mma(lower_source):
    program = lower_source(TCGEN)
    alloc = only(program, "TcgenAlloc")
    assert (alloc.cta_group, alloc.exclusive, const(program, alloc.ncols)) == (1, False, 128)
    assert alloc.may_block
    dealloc, relinquish = only(program, "TcgenDealloc"), only(program, "TcgenRelinquish")
    assert dealloc.cta_group == 1 and relinquish.cta_group == 1 and not dealloc.may_block

    encoders = [op_key(program, i) for i in all_of(program, "Ptx") if op_key(program, i).name.startswith("tirx.cuda.tcgen05")]
    assert [k.name for k in encoders] == ["tirx.cuda.tcgen05_encode_matrix_descriptor"] * 2 + [
        "tirx.cuda.tcgen05_encode_instr_descriptor"]
    assert encoders[2].mods == ("arg1=float32", "arg2=float16", "arg3=float16")
    # The descriptor out-parameters bind directly to the promoted locals.
    instr_desc = [i for i in all_of(program, "Ptx") if op_key(program, i) == encoders[2]][0]
    assert program.regs[instr_desc.dsts[0].index].name == "idesc"

    mma = only(program, "TcgenMma")
    assert (mma.kind, mma.cta_group, mma.ws, mma.block_scale, mma.sparse_meta) == ("F16", 1, False, None, None)
    assert list(mma.a) == ["Smem"] and len(mma.disable_output_lane) == 4
    assert mma.idesc == instr_desc.dsts[0]
    assert program.regs[mma.enable_input_d.index].ty == pb.Ty("Pred") if isinstance(mma.enable_input_d, pb.Reg) \
        else program.consts[mma.enable_input_d.index][0] == pb.Ty("Pred")

    commit = only(program, "TcgenCommit")
    assert (commit.space, commit.cta_group, commit.multicast) == ("SharedCluster", 1, None)

    ld = only(program, "TcgenLd")
    assert (ld.shape, ld.num, ld.pack, ld.red) == ("S32x32b", 4, False, None)
    assert [program.regs[r.index].name for r in ld.dsts] == ["regs"] * 4
    wait = only(program, "TcgenWait")
    assert wait.st is False and wait.may_block


def test_tmem_view_direct_access(lower_source):
    """Phase 3 ruling: a TMEM DeclBuffer is a Space::Tmem Buf; Load/Store use dense lane x column offsets."""
    program = lower_source('''
@T.prim_func
def k(out: T.Buffer((128, 4), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    T.cta_id([1])
    T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=T.TileLayout(T.S[(128, 4):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=0)
    for col in range(4):
        tmem[warp * 32 + lane, col] = T.Cast("uint32", col)
    for col in range(4):
        out[warp * 32 + lane, col] = tmem[warp * 32 + lane, col]
''')
    tmem = next(b for b in program.buffers if b.name == "tmem")
    assert (tmem.space, tmem.dtype, tmem.base) == ("Tmem", pb.Ty("U32"), 0)
    assert tmem.shape == (pb.DimExpr.const(128), pb.DimExpr.const(4))
    buf = program.buffers.index(tmem)
    store = next(i for i in all_of(program, "Store") if i.buf == buf)
    load = next(i for i in all_of(program, "Load") if i.buf == buf)
    assert store.ty == load.ty == pb.Ty("U32")
    # offset = lane * 4 + col (dense addressing), computed by a Mul by the column span then an Add.
    offset = next(i for i in program.code if store.offset in i.writes())
    assert offset.variant == "Binary" and offset.op == "Add"
    row = next(i for i in program.code if offset.a in i.writes())
    assert (row.op, const(program, row.b)) == ("Mul", 4)
    assert program.requirements.implicit_tmem      # views without tcgen05.alloc
    assert not all_of(program, "AddrOf")            # no addresses of TMEM


def test_subword_tmem_view_packs_elements_per_cell(lower_source):
    """Contract item 28: a u16 TMEM view counts its columns in cells; the offset is in elements."""
    program = lower_source('''
@T.prim_func
def k(output: T.Buffer((128,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    low = T.decl_buffer((128, 2), "uint16", scope="tmem", layout=T.TileLayout(T.S[(128, 2):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=5)
    word = T.decl_buffer((128, 2), "uint32", scope="tmem", layout=T.TileLayout(T.S[(128, 2):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=5)
    low[warp * 32 + lane, 1] = T.uint16(13124)
    output[warp * 32 + lane] = word[warp * 32 + lane, 0]
''')
    by_name = {b.name: b for b in program.buffers}
    assert by_name["low"].dtype == pb.Ty("U16") and by_name["low"].shape[1] == pb.DimExpr.const(1)
    assert by_name["word"].shape[1] == pb.DimExpr.const(2)
    store = next(i for i in all_of(program, "Store") if program.buffers[i.buf].name == "low")
    assert store.ty == pb.Ty("U16")


def test_replicated_tmem_view_access_names_the_contract_reason(lower_source):
    """Contract item 29: `Unsupported { reason: "tmem_replicated_view: <buffer>" }`."""
    program = lower_source('''
@T.prim_func
def k(output: T.Buffer((128,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    rep = T.decl_buffer((128,), "uint32", scope="tmem", layout=T.TileLayout(T.S[(128,):(1 @ Axis.TLane)] + T.R[(2,):(1 @ Axis.TCol)]), allocated_addr=0)
    output[warp * 32 + lane] = rep[warp * 32 + lane]
''', strict=False)
    assert any("tmem_replicated_view: rep" in reason for reason in program.unsupported)
