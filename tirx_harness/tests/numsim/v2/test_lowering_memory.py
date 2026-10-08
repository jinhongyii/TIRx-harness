"""Memory family: PTX ld/st, shared pool views, swizzles, atomics, local promotion."""

from __future__ import annotations

from tirx_harness.numsim.v2.lowering import program_builder as pb

from ._program import all_of, const, definition, only, op_key

POOL_KERNEL = '''
@T.prim_func
def k(a: T.Buffer((128,), "float32"), out: T.Buffer((128,), "float32")):
    T.func_attr({"tirx.cuda_arch": "sm_100a"})
    T.attr({"tirx.device_entry": T.bool(True), "tirx.dyn_smem_bytes": 1024})
    lane = T.lane_id([32])
    T.warp_id([1])
    pool = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    mbar = T.decl_buffer((2,), "uint64", data=pool.data, elem_offset=0, scope="shared.dyn", align=8)
    tile = T.decl_buffer((128,), "float32", data=pool.data, elem_offset=4, scope="shared.dyn", align=16)
    v: T.float32
    T.ptx.ld(v, T.address_of(a[lane]), "", "acquire", "gpu", "global", "", "", "", "", "", "f32", "")
    T.ptx.st(T.cuda.cvta_generic_to_shared(T.address_of(tile[lane])), v, "", "", "", "shared", "", "", "", "f32", "")
    out[lane] = tile[lane]
'''


def test_shared_pool_views_and_ptx_buffer_form(lower_source):
    program = lower_source(POOL_KERNEL)
    names = [b.name for b in program.buffers]
    pool, mbar, tile = (program.buffers[names.index(n)] for n in ("pool", "mbar", "tile"))
    assert (pool.space, pool.view_of, pool.byte_len) == ("Shared", None, pb.DimExpr.const(1024))
    assert (mbar.view_of, mbar.base, mbar.dtype) == (names.index("pool"), 0, pb.Ty("U64"))
    # elem_offset=4 float32 elements -> byte 16 of the pool.
    assert (tile.view_of, tile.base, tile.byte_len) == (names.index("pool"), 16, pb.DimExpr.const(512))
    assert program.topology.static_smem_bytes == 1024

    ld, st = all_of(program, "Load")[0], all_of(program, "Store")[0]
    # ptx.ld on address_of(a[lane]) keeps the buffer-relative form with its modifiers.
    assert (ld.buf, ld.ty, ld.sem, ld.scope) == (names.index("a"), pb.Ty("F32"), "Acquire", "Gpu")
    # ptx.st through cvta(address_of(tile[lane])) with space=shared -> Store into the view.
    assert (st.buf, st.ty) == (names.index("tile"), pb.Ty("F32"))
    assert definition(program, st.value).variant == "Mov"


def test_raw_address_load_and_vector_unpack(lower_source):
    program = lower_source('''
@T.prim_func
def k(p: T.Buffer((256,), "uint32"), out: T.Buffer((128,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    r = T.alloc_local((4,), "uint32")
    T.ptx.ld(r[0], r[1], r[2], r[3], T.ptr_byte_offset(T.address_of(p[0]), lane * 16, T.type_annotation("uint32")), "", "", "", "global", "", "", "", "", "", "v4", "u32", "")
    out[lane] = r[0] + r[3]
''')
    load = only(program, "LoadAddr")
    assert (load.ty, load.space) == (pb.Ty("U32", 4), "Global")
    unpack = next(i for i in all_of(program, "Ptx") if op_key(program, i).name == "numsim.unpack")
    assert unpack.srcs == [load.dst] and len(unpack.dsts) == 4
    assert op_key(program, unpack).mods == ("ty=U32x4",)
    # Lanes land in the promoted registers of r[0..3].
    movs = [i for i in all_of(program, "Mov") if i.src in unpack.dsts]
    assert [program.regs[m.dst.index].name for m in movs] == ["r"] * 4


def test_compose_layout_swizzle_becomes_xor_arithmetic(lower_source):
    program = lower_source('''
@T.prim_func
def k(out: T.Buffer((64, 64), "float16")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    s = T.alloc_shared((64, 64), "float16", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[4096:1])))
    s[lane, 3] = T.float16(1.0)
    out[lane, 3] = s[lane, 3]
''')
    store = next(i for i in all_of(program, "Store") if program.buffers[i.buf].name == "s")
    xors = [i for i in all_of(program, "Binary") if i.op == "Xor"]
    assert xors, "swizzled offsets are computed with Xor"
    # The physical offset of the store is derived from the swizzle arithmetic.
    seen, frontier = set(), [store.offset]
    while frontier:
        reg = frontier.pop()
        if not isinstance(reg, pb.Reg) or reg.index in seen:
            continue
        seen.add(reg.index)
        writer = definition(program, reg)
        frontier.extend(writer.operands())
    assert any(x.dst.index in seen for x in xors)


def test_atomics_and_escaped_locals(lower_source):
    program = lower_source('''
@T.prim_func
def k(counter: T.Buffer((1,), "int32"), out: T.Buffer((32,), "int32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    old: T.int32
    T.ptx.atom(old, T.address_of(counter[0]), T.int32(1), "relaxed", "gpu", "global", "add", "", "s32", "")
    pair = T.alloc_local((2,), "float32")
    h: T.uint32
    pair[0] = T.float32(1.0)
    pair[1] = T.float32(2.0)
    T.cuda.float22half2(T.address_of(h), T.address_of(pair[0]))
    out[lane] = old
''')
    atom = only(program, "Atom")
    assert (atom.op, atom.ty, atom.space, atom.sem, atom.scope) == ("Add", pb.Ty("S32"), "Global", "Relaxed", "Gpu")
    assert const(program, atom.value) == 1 and atom.dst is not None
    # Both locals escape through address_of -> per-lane Local memory, not registers.
    locals_ = {b.name: b for b in program.buffers if b.space == "Local"}
    assert set(locals_) == {"pair", "h"}
    cvt = next(i for i in all_of(program, "Ptx") if op_key(program, i).name.startswith("tirx.cuda.float22half2"))
    assert cvt.dsts and cvt.srcs
    store = all_of(program, "StoreAddr")[-1]
    assert store.ty == pb.Ty("F16", 2)


def test_constant_and_dynamic_local_indexing(lower_source):
    program = lower_source('''
@T.prim_func
def k(out: T.Buffer((32,), "float32"), sel: T.Buffer((32,), "int32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    acc = T.alloc_local((8,), "float32")
    for i in T.unroll(8):
        acc[i] = T.Cast("float32", i)
    out[lane] = acc[3] + acc[sel[lane]]
''')
    # T.unroll stays a loop (decision 2) and its dynamic index is a register store.
    assert only(program, "LoopBegin")
    store = only(program, "StoreRegIndexed")
    load = only(program, "LoadRegIndexed")
    assert store.len == load.len == 8 and store.base == load.base
    assert program.regs[store.base.index].name == "acc"
    assert all(b.space != "Local" for b in program.buffers)


VECTOR_VIEW_KERNEL = '''
@T.prim_func
def k(state: T.Buffer((1024,), "uint16"), out: T.Buffer((128,), "uint32x4"), out2: T.Buffer((128,), "uint16")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    wide = T.decl_buffer((128,), "uint32x4", data=state.data, scope="global")
    halves = T.decl_buffer((512,), "uint16", data=state.data, scope="global")
    out[lane] = wide[lane]
    out2[lane] = halves[lane]
'''


def test_vector_view_offsets_count_scalar_elements(lower_source):
    """V2C-11: `Load.offset` counts `dtype.elem`, so a `uint32x4` element is 4 units."""
    program = lower_source(VECTOR_VIEW_KERNEL)
    names = [b.name for b in program.buffers]
    wide = names.index("wide")
    load = next(i for i in all_of(program, "Load") if i.buf == wide)
    assert load.ty == pb.Ty("U32", 4)
    scale = definition(program, load.offset)
    assert (scale.variant, scale.op, const(program, scale.b)) == ("Binary", "Mul", 4)


def test_dtype_changing_view_is_its_own_logical_buffer(lower_source):
    """W5-9: a dtype-changing view is a new identity; same-dtype views keep the root."""
    program = lower_source(VECTOR_VIEW_KERNEL)
    names = [b.name for b in program.buffers]
    site = lambda instr: program.site_of(program.code.index(instr)).buffer  # noqa: E731
    loads = {names[i.buf]: i for i in all_of(program, "Load")}
    assert site(loads["wide"]) == "wide"
    assert site(loads["halves"]) == "state"
    assert all(b.name for b in program.buffers)
