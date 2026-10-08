"""Sync and async families: mbarrier, barriers, fences, elect, TMA + host prelude, bulk groups."""

from __future__ import annotations

import json

from tirx_harness.numsim.v2.lowering import program_builder as pb

from ._program import all_of, const, definition, only

PIPELINE = '''
@T.prim_func
def k(a: T.Buffer((256, 64), "float16")):
    T.func_attr({"tirx.cuda_arch": "sm_100a"})
    v: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.tensormap_encode_tiled(v, a.data, 64, 256, 128, 64, 128, 1, 1, descriptor_dtype="float16", rank=2, interleave=0, swizzle=3, l2_promotion=2, oob_fill=0, force_cu_dtype=-1)
    with T.attr({"tirx.device_entry": T.bool(True)}):
        T.cta_id([2])
        cta, _y = T.cta_id_in_cluster([2, 1], preferred=[2, 1])
        warp = T.warp_id([4])
        lane = T.lane_id([32])
        pool = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
        full = T.decl_buffer((2,), "uint64", data=pool.data, elem_offset=0, scope="shared.dyn", align=8)
        tile = T.decl_buffer((2, 128, 64), "float16", data=pool.data, elem_offset=512, scope="shared.dyn", align=1024)
        if warp == 0:
            if T.cuda.elect_sync() != T.uint32(0):
                T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(full[0])), T.uint32(1), "init", "", "shared", "b64", "")
        T.ptx.fence("mbarrier_init", "release", "cluster", "")
        T.cuda.cluster_sync()
        if warp == 0:
            if T.cuda.elect_sync() != T.uint32(0):
                T.ptx.cp(T.cuda.cvta_generic_to_shared(T.address_of(tile[0, 0, 0])), T.reinterpret(T.handle().ty, T.address_of(v)), 0, cta * 128, T.cuda.cvta_generic_to_shared(T.address_of(full[0])), "async", "bulk", "tensor", "2d", "shared::cluster", "global", "", "mbarrier::complete_tx::bytes", "", "cta_group::2", "", "")
                T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(full[0])), T.uint32(16384), "arrive", "expect_tx", "", "", "shared::cluster", "b64", "")
        T.cuda.mbarrier_wait(T.address_of(full[0]), 0)
        T.ptx.fence("proxy", "async", "shared::cta", "")
        T.cuda.cta_sync()
'''


def test_host_prelude_becomes_implicit_tensor_map_slot(lower_source):
    program = lower_source(PIPELINE)
    kinds = [(s.name, s.kind) for s in program.host_abi]
    assert kinds == [("a", "Buffer"), ("v.tmap", "TensorMap")]  # V2C-6: own slot identity
    # The prelude var still binds (caller override) while no other slot is named `v`.
    assert program.host_abi[1].aliases == ("v",)
    spec = program.host_abi[1].tensor_map
    assert (spec.dtype, spec.rank, spec.swizzle, spec.l2_promotion) == ("F16", 2, 3, 2)
    assert spec.global_dim == (pb.DimExpr.const(64), pb.DimExpr.const(256))
    assert spec.global_stride == (pb.DimExpr.const(128),)
    assert [d.value for d in spec.box_dim] == [64, 128] and [d.value for d in spec.element_stride] == [1, 1]
    assert program.host_abi[1].implicit_base == 0
    tmap_buf = program.buffers[program.host_abi[1].buf]
    assert (tmap_buf.space, tmap_buf.param_slot, tmap_buf.byte_len) == ("Param", 1, pb.DimExpr.const(128))
    assert program.topology.cluster == (2, 1, 1) and program.topology.block == (128, 1, 1)


def test_tma_load_and_mbarrier_protocol(lower_source):
    program = lower_source(PIPELINE)
    init = only(program, "MbarInit")
    assert (init.space, const(program, init.count), init.layout_v1) == ("Shared", 1, False)
    fence_kinds = [i.kind for i in all_of(program, "Fence")]
    assert fence_kinds == ["MbarrierInit", {"ProxyAsync": "Shared"}]

    tma = only(program, "Tma")
    assert (tma.dir, tma.mode, tma.cta_group, tma.smem_space) == ("Load", "Tile", 2, "SharedCluster")
    assert len(tma.coords) == 2 and tma.multicast is None
    assert list(tma.completion) == ["Mbarrier"] and tma.completion["Mbarrier"]["space"] == "Shared"
    # W5-7: sites name the logical buffer (the view), never the shared.dyn pool.
    site = lambda instr: program.site_of(program.code.index(instr)).buffer  # noqa: E731
    assert site(tma) == "tile" and site(only(program, "MbarWait")) == "full"
    tmap_addr = next(i for i in all_of(program, "AddrOf") if i.dst == tma.tmap)
    assert program.buffers[tmap_addr.buf].space == "Param"
    # V2C-9: the map operand is a generic address (param aperture), never Global.
    assert tma.tmap_space == "Generic"

    arrive = only(program, "MbarArrive")
    assert (arrive.space, const(program, arrive.expect_tx), arrive.count, arrive.sem, arrive.scope) == (
        "SharedCluster", 16384, None, "Release", "Cta")
    wait = only(program, "MbarWait")
    assert (wait.space, wait.sem, wait.scope) == ("Generic", "Acquire", "Cta")
    assert list(wait.phase) == ["Parity"]
    assert wait.may_block and not arrive.may_block

    # The elect-gated regions carry the elect flag.
    elect_ifs = [i for i in all_of(program, "If") if i.elect]
    assert len(elect_ifs) == 2

    arrive_c, wait_c = only(program, "ClusterArrive"), only(program, "ClusterWait")
    assert (arrive_c.sem, arrive_c.aligned, wait_c.acquire) == ("Release", True, True)
    barrier = only(program, "Barrier")
    assert (barrier.kind, const(program, barrier.id), barrier.count) == ("Sync", 0, None)


def test_bulk_groups_cp_async_and_named_barriers(lower_source):
    program = lower_source('''
@T.prim_func
def k(g: T.Buffer((1024,), "float32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    warp = T.warp_id([8])
    lane = T.lane_id([32])
    s = T.alloc_shared((1024,), "float32")
    T.ptx.cp(T.cuda.cvta_generic_to_shared(T.address_of(s[lane * 4])), T.address_of(g[lane * 4]), 16, T.uint32(16), "async", "cg", "shared", "global", "", "", "")
    T.ptx.cp("async", "commit_group", "")
    T.ptx.cp(0, "async", "wait_group", "")
    T.ptx.cp(T.address_of(g[0]), T.cuda.cvta_generic_to_shared(T.address_of(s[0])), T.uint32(4096), "async", "bulk", "", "", "global", "shared::cta", "bulk_group", "", "", "", "")
    T.ptx.cp("async", "bulk", "commit_group", "")
    T.ptx.cp(0, "async", "bulk", "wait_group", "read", "")
    T.ptx.bar(T.uint32(1), T.uint32(128), "", "sync", "")
    T.cuda.warpgroup_sync(2)
''')
    cp = only(program, "CpAsync")
    assert cp.cp_size == 16 and program.consts[cp.src_size.index][1] == 16
    commits = [i.domain for i in all_of(program, "AsyncCommit")]
    waits = [(i.domain, i.n, i.read) for i in all_of(program, "AsyncWait")]
    assert commits == ["CpAsync", "Bulk"] and waits == [("CpAsync", 0, False), ("Bulk", 0, True)]
    bulk = only(program, "BulkCopy")
    assert (bulk.dst_space, bulk.src_space, bulk.completion, bulk.reduce) == ("Global", "Shared", "Group", None)
    barriers = [(i.kind, const(program, i.id), const(program, i.count)) for i in all_of(program, "Barrier")]
    assert barriers == [("Sync", 1, 128), ("Sync", 2, 128)]


def test_unqualified_mbarrier_operand_is_generic_or_shared_cluster(lower_source):
    """W2-13 ruling: no state space -> 64-bit Generic, 32-bit SharedCluster, never Shared."""
    program = lower_source('''
@T.prim_func
def k():
    T.attr({"tirx.device_entry": T.bool(True), "tirx.dyn_smem_bytes": 64})
    lane = T.lane_id([32])
    T.warp_id([1])
    pool = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    full = T.decl_buffer((2,), "uint64", data=pool.data, elem_offset=0, scope="shared.dyn", align=8)
    generic = T.reinterpret(T.handle().ty, T.reinterpret("uint64", T.address_of(full[0])))
    T.ptx.mbarrier(generic, T.uint32(64), "arrive", "expect_tx", "", "", "", "b64", "")
    rem = T.alloc_local((1,), "uint64")
    T.ptx.mapa.shared__cluster.u64(rem[0], full.ptr_to([1]), T.uint32(1))
    T.ptx.mbarrier.arrive.expect_tx.b64(rem[0], T.uint32(64))
''')
    # (TVM rejects an unqualified 32-bit operand at parse time; lowering maps
    # one, e.g. a raw `mapa` u32 result, to SharedCluster.)
    wide, remote = all_of(program, "MbarArrive")
    assert wide.space == "Generic"
    # A `mapa.shared::cluster` result is a cluster-window address even in a u64.
    assert remote.space == "SharedCluster"
    assert definition(program, wide.mbar).variant != "Cvta"  # no forced shared::cta


def test_explicit_tensor_map_param_keeps_its_declared_name(lower_source):
    """Only implicit prelude maps get the `.tmap` suffix (V2C-6); a declared map param keeps its name."""
    program = lower_source('''
@T.prim_func
def k(a: T.Buffer((64,), "float32"), tensor_map: T.handle("tensormap")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    a[lane] = T.float32(0)
''')
    assert [(s.name, s.kind) for s in program.host_abi][:2] == [("a", "Buffer"), ("tensor_map", "TensorMap")]


def test_runtime_tensor_map_box_is_a_param_expression():
    """Contract item 30: box_dim / element_stride are DimExprs; runtime prologue values stay symbolic."""
    from tests.numsim.integration.test_host_prelude import host_encoded_dynamic_integer_tensor_map
    from tirx_harness.numsim.v2.lowering import lower

    program = lower(host_encoded_dynamic_integer_tensor_map)
    spec = next(s.tensor_map for s in program.host_abi if s.tensor_map is not None)
    assert not spec.box_dim[0].is_const
    assert "Param" in json.dumps(spec.box_dim[0].to_json())
    assert spec.element_stride[0] == pb.DimExpr.const(1)
