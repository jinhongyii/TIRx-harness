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
    assert site(loads["state"]) == "state"  # same-dtype global view: addressed in its root
    assert all(b.name for b in program.buffers)


def _uninit_kernel(body: str) -> str:
    return f'''
@T.prim_func
def k(out: T.Buffer((32,), "float32"), idx: T.Buffer((32,), "int32")):
    T.attr({{"tirx.device_entry": T.bool(True)}})
    lane = T.lane_id([32])
    T.warp_id([1])
    a = T.alloc_local((4,), "float32")
{body}
'''


def _local_buffers(program: pb.Program) -> list[str]:
    return [b.name for b in program.buffers if b.space in ("Local", "Reg")]


def test_possibly_uninitialized_locals_stay_in_tracked_memory(lower_source):
    """V2C-19/20: a local read before some write lives in Local memory (validity tracked)."""
    read_first = lower_source(_uninit_kernel("    out[lane] = a[0]"))
    assert _local_buffers(read_first) == ["a"]
    conditional = lower_source(_uninit_kernel(
        "    if lane < 4:\n        a[0] = T.float32(1)\n    out[lane] = a[0]"))
    assert _local_buffers(conditional) == ["a"]
    partial = lower_source(_uninit_kernel(
        "    a[0] = T.float32(1)\n    out[lane] = a[idx[lane]]"))
    assert _local_buffers(partial) == ["a"]


def test_provably_initialized_locals_are_promoted(lower_source):
    straight = lower_source(_uninit_kernel(
        "    a[0] = T.float32(1)\n    out[lane] = a[0]"))
    assert _local_buffers(straight) == []
    loop_init = lower_source(_uninit_kernel(
        "    for i in range(4):\n        a[i] = T.float32(0)\n    out[lane] = a[idx[lane]]"))
    assert _local_buffers(loop_init) == [] and all_of(loop_init, "LoadRegIndexed")
    both_branches = lower_source(_uninit_kernel(
        "    if lane < 4:\n        a[0] = T.float32(1)\n    else:\n        a[0] = T.float32(2)\n"
        "    out[lane] = a[0]"))
    assert _local_buffers(both_branches) == []


def test_sub_byte_view_footprint_and_view_of_view_base(lower_source):
    """W4 (mqa_logits_fp4): an E2M1 view takes numel/2 bytes, and a view of a view
    is placed relative to its parent (`elem_offset` counts from the shared data pointer)."""
    program = lower_source('''
@T.prim_func
def k(out: T.Buffer((32,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True), "tirx.dyn_smem_bytes": 2048})
    lane = T.lane_id([32])
    T.warp_id([1])
    pool = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    q = T.decl_buffer((2, 128), "float4_e2m1fn", data=pool.data, elem_offset=0, scope="shared.dyn", align=16)
    sf = T.decl_buffer((32,), "uint32", data=pool.data, elem_offset=32, scope="shared.dyn", align=16)
    sf2 = T.decl_buffer((2, 16), "uint32", data=sf.data, elem_offset=32, scope="shared.dyn", align=16)
    out[lane] = sf[lane] + sf2[lane // 16, lane % 16]
''')
    by_name = {b.name: (i, b) for i, b in enumerate(program.buffers)}
    _, q = by_name["q"]
    sf_index, sf = by_name["sf"]
    _, sf2 = by_name["sf2"]
    assert q.byte_len == pb.DimExpr.const(128)  # 256 x 4 bits
    assert sf.base == 128 and sf.byte_len == pb.DimExpr.const(128)
    # sf2 aliases sf exactly: base 0 within sf, not 128 past it.
    assert (sf2.view_of, sf2.base) == (sf_index, 0)


def test_register_layout_local_indexes_by_register_and_asserts_owner(lower_source):
    """A fragment layout (laneid, m): storage offset is `m`; the owning lane is asserted."""
    program = lower_source('''
@T.prim_func
def k(out: T.Buffer((64,), "float32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    frag = T.alloc_buffer((64,), "float32", scope="local", layout=T.TileLayout(T.S[(32, 2):(1 @ Axis.laneid, 1)]))
    for i in range(2):
        frag[lane * 2 + i] = T.float32(1)
    for i in range(2):
        out[lane * 2 + i] = frag[lane * 2 + i]
''')
    assert not program.unsupported
    assert len(all_of(program, "Assert")) == 2


def test_predicated_vector_atomic_is_guarded():
    """W4-12: `@p atom.v2.f32` adds only where the guard holds (the Atom sits inside the If)."""
    from tests.numsim.v2._kernels import atomic_kernel
    from tirx_harness.numsim.v2.lowering import lower

    program = lower(atomic_kernel("atom", 2, "global"))
    variants = [i.variant for i in program.code]
    atom = variants.index("Atom")
    # `if warp == 0` plus the instruction guard, both open at the Atom.
    open_ifs = 0
    for variant in variants[:atom]:
        open_ifs += variant == "If"
        open_ifs -= variant == "EndIf"
    assert open_ifs == 2
    assert program.code[atom].ty == pb.Ty("F32", 2)


def test_vector_red_packs_pieces_that_tile_the_access_type():
    """`red.v2.bf16x2` packs two bf16x2 pieces into bf16x4 (pack tiling rule)."""
    from tests.numsim.v2._kernels import packed_bf16_vector_reduction
    from tirx_harness.numsim.v2.lowering import lower

    program = lower(packed_bf16_vector_reduction.func)
    atom = only(program, "Atom")
    assert atom.ty == pb.Ty("BF16", 4)
    pack = definition(program, atom.value)
    assert op_key(program, pack).name == "numsim.pack"
    def ty(operand):
        return program.consts[operand.index][0] if isinstance(operand, pb.Const) else program.regs[operand.index].ty

    assert [ty(s) for s in pack.srcs] == [pb.Ty("BF16", 2), pb.Ty("BF16", 2)]


def test_dtype_changing_view_over_offset_global_view_applies_the_offset_once():
    """W2: `o.view("uint64")` of `o = decl_buffer(data=out.data, elem_offset=off)` addresses
    root + off * 2 + index * 8 (elem_offset counts from the data pointer)."""
    import numpy as np
    import tvm
    from tvm.script import tirx as T

    from tirx_harness.numsim import v2

    kernel = tvm.script.from_source('''
@T.prim_func
def k(out: T.Buffer((256,), "bfloat16"), off: T.int32):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    o = T.decl_buffer((64,), "bfloat16", data=out.data, elem_offset=off, scope="global")
    w = T.decl_buffer((16,), "uint64", data=o.data, elem_offset=off // 4, scope="global")
    if lane < 16:
        w[lane] = T.uint64(0x0001000100010001) * T.Cast("uint64", lane + 1)
''', {"T": T})
    result = v2.Engine().run(v2.transpile(kernel), {"out": np.zeros(256, np.uint16), "off": 128})
    out = result.outputs["out"].view(np.uint16)
    assert np.nonzero(out)[0].tolist() == list(range(128, 192))
    assert out[128] == 1 and out[191] == 16


def test_packed_eight_conversions_read_their_first_pointer():
    """`cuda_{float8tohalf8,half8tofloat8}(void* src, void* dst)`: source first
    (unlike `float22half2(dst, src)`), as TVM's builtins and legacy define them."""
    from tests.numsim.v2._kernels import pointer_conversions_and_descriptor
    from tirx_harness.numsim.v2.lowering import lower

    program = lower(pointer_conversions_and_descriptor)

    def buffer_of(addr):
        instr = definition(program, addr)
        while instr.variant != "AddrOf":
            instr = definition(program, next(o for o in instr.operands() if isinstance(o, pb.Reg)))
        return program.buffers[instr.buf].name

    loads = [(i.ty, buffer_of(i.addr)) for i in all_of(program, "LoadAddr")]
    stores = [(i.ty, buffer_of(i.addr)) for i in all_of(program, "StoreAddr")]
    assert (pb.Ty("F32", 8), "source") in loads and (pb.Ty("F16", 8), "half") in stores
    assert (pb.Ty("F16", 8), "half") in loads and (pb.Ty("F32", 8), "roundtrip") in stores


def test_cross_owner_fragment_copy_is_transported_through_shared_scratch():
    """A copy between register fragments with different thread layouts stages the
    source through a shared scratch, synchronizes the warpgroup, then each
    destination owner writes its own elements (no cross-thread register writes)."""
    from tests.numsim.v2._kernels import _copy_cross_warp_owner_remap
    from tirx_harness.numsim.v2.lowering import lower

    program = lower(_copy_cross_warp_owner_remap)
    scratch = [b for b in program.buffers if b.name.endswith(".transport")]
    assert len(scratch) == 1 and scratch[0].space == "Shared"
    barriers = all_of(program, "Barrier")
    assert len(barriers) == 2 and all(const(program, b.id) == 8 for b in barriers)
    assert not any(i.variant == "Unsupported" for i in program.code)


def test_legacy_mma_fill_and_store_follow_the_lane_register_layout():
    """`tirx.mma_fill` zeroes the fragment; `tirx.mma_store` writes element `id` of lane `l`
    to row 8*((id%4)//2) + l//4, col 8*(id//4) + 2*(l%4) + id%2 (legacy emit/matrix.rs)."""
    import numpy as np

    from tests.numsim.v2._kernels import mma_fragment_fill_and_store
    from tirx_harness.numsim import v2

    result = v2.Engine().run(v2.transpile(mma_fragment_fill_and_store), {
        "filled": np.ones((2, 32, 8), dtype=np.float32), "stored": np.zeros((2, 16, 16), dtype=np.float32)})
    assert not result.outputs["filled"].any()
    stored = result.outputs["stored"]
    for row, col in ((0, 0), (9, 3), (15, 15)):
        lane, local_id = 4 * (row % 8) + (col % 8) // 2, 4 * (col // 8) + 2 * (row // 8) + col % 2
        assert stored[0, row, col] == lane * 100 + local_id


def test_shared_scope_view_over_a_mapa_result_reads_the_cluster_window():
    """A `decl_buffer(scope="shared", data=<mapa.shared::cluster result>)` is addressed in
    shared::cluster space (32-bit window address), not as a generic pointer (W2)."""
    from tests.numsim.v2._kernels import canonical_mapa
    from tirx_harness.numsim.v2.lowering import lower

    kernel = canonical_mapa
    program = lower(kernel)
    reads = [i for i in all_of(program, "LoadAddr") if i.space != "Generic"]
    assert reads and all(i.space == "SharedCluster" for i in reads)


def _window_kernel(arch: str | None, pool_bytes: int, view_offset: int, view_elems: int) -> str:
    attr = f'T.func_attr({{"tirx.cuda_arch": "{arch}"}})\n    ' if arch else ""
    return f'''
@T.prim_func
def k():
    {attr}T.attr({{"tirx.device_entry": T.bool(True)}})
    lane = T.lane_id([32])
    T.warp_id([1])
    pool = T.alloc_buffer(({pool_bytes},), "uint8", scope="shared.dyn", align=1024)
    tile = T.decl_buffer(({view_elems},), "bfloat16", data=pool.data, elem_offset={view_offset}, scope="shared.dyn")
    tile[lane] = T.bfloat16(1)
'''


def test_view_overrunning_its_pool_does_not_grow_the_shared_window(lower_source):
    # cudnn_sm100_flex_attention_forward_hd256: a 230272-byte pool whose last
    # bf16 staging view is declared 32768 elements at byte 196992 (ends at
    # 262528) but only touches its first half. The CTA window is the allocation.
    program = lower_source(_window_kernel("sm_100a", 230272, 98496, 32768))
    pool = next(b for b in program.buffers if b.name == "pool")
    assert pool.byte_len == pb.DimExpr.const(230272)
    assert program.topology.static_smem_bytes == 230272


def test_shared_window_above_the_sm100_per_cta_capacity_fails_closed(lower_source):
    capacity = 227 * 1024
    assert lower_source(_window_kernel("sm_100a", capacity, 0, 32)).topology.static_smem_bytes == capacity
    for arch in ("sm_100a", "sm_103a", "sm_100f"):
        program = lower_source(_window_kernel(arch, capacity + 16, 0, 32), strict=False)
        assert any(f"above the {capacity}-byte per-CTA capacity of {arch}" in u for u in program.unsupported), \
            program.unsupported
    # Targets without a modeled capacity (sm_107a, no tirx.cuda_arch) are not checked.
    for arch in ("sm_107a", None):
        assert lower_source(_window_kernel(arch, 300000, 0, 32)).topology.static_smem_bytes == 300000


def test_explicit_alignment_places_a_swizzled_shared_backing(lower_source):
    """An explicit ``align=`` is the placement (legacy cvta alignment test);
    without one a swizzled operand starts on its swizzle repeat (1024 B here)."""
    from tests.numsim.support.kernels import shared_virtual_swizzled_backing_alignment
    from tirx_harness.numsim.v2.lowering import lower

    program = lower(shared_virtual_swizzled_backing_alignment)
    bases = {b.name: (b.base, b.align) for b in program.buffers if b.space == "Shared"}
    assert bases["first"][0] == 0
    assert bases["second"] == (128, 128)


def test_pool_max_bytes_sizes_a_zero_extent_pool_and_fails_closed_when_malformed(lower_source):
    """``AttrStmt(<alloc>.data, "tirx.pool_max_bytes", N)`` (legacy analyze/memory.rs):
    a zero-extent pool is exactly N bytes (no growth from its views); a negative or
    conflicting capacity is rejected."""
    head = '''
@T.prim_func
def k(output: T.Buffer((1,), "uint32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared")
'''
    sized = lower_source(head + '''    T.attr(storage.data, "tirx.pool_max_bytes", 16)
    alias = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared")
    output[0] = alias[0]
''')
    pool = next(b for b in sized.buffers if b.name == "storage")
    assert pool.byte_len == pb.DimExpr.const(16) and sized.topology.static_smem_bytes == 16
    for body, reason in (
        ('    T.attr(storage.data, "tirx.pool_max_bytes", -1)\n    output[0] = 0\n', "cannot be negative"),
        (
            '    T.attr(storage.data, "tirx.pool_max_bytes", 16)\n'
            '    T.attr(storage.data, "tirx.pool_max_bytes", 32)\n    output[0] = 0\n',
            "conflicting capacities 16 and 32",
        ),
    ):
        program = lower_source(head + body, strict=False)
        assert any(reason in u for u in program.unsupported), program.unsupported
