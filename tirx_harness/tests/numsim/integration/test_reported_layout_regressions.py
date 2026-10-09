from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim, racecheck, synccheck
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.tile_primitive.tma_utils import (
    SwizzleMode,
    mma_shared_layout,
)
from tvm.tirx.lang.pipeline import TCGen05Bar, TMABar
from tvm.tirx.layout import (
    R,
    S,
    TileLayout,
    laneid,
    tmem_datapath_layout,
    wg_local_layout,
)


LANES = 32
_LANE_OWNER_LAYOUT = TileLayout(S[LANES : 1 @ laneid])


@T.prim_func
def lane_owner_warp_sum(
    source: T.Buffer((LANES,), "float32"),
    output: T.Buffer((LANES,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([LANES])
    source_scalar = T.alloc_buffer((1,), "float32", scope="local")
    result_scalar = T.alloc_buffer((1,), "float32", scope="local")
    source_scalar[0] = source[lane]
    source_tile = source_scalar.view(LANES, layout=_LANE_OWNER_LAYOUT)
    result_tile = result_scalar.view(
        1,
        layout=TileLayout(S[1:1] + R[LANES : 1 @ laneid]),
    )
    Tx.warp.sum(result_tile, source_tile, thread_reduce=True)
    output[lane] = result_scalar[0]


@T.prim_func
def one_register_m64_fragment(output: T.Buffer((128,), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    thread = T.thread_id_in_wg([128])
    fragment = T.alloc_tcgen05_ldst_frag("16x64b", (64, 2), "float32")
    local = fragment.local(1)
    local[0] = T.cast(thread, "float32")
    output[thread] = local[0]


@T.prim_func
def explicit_shared_strides_roundtrip(output: T.Buffer((64,), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    thread = T.thread_id_in_wg([128])
    pool = T.SMEMPool()
    scratch = pool.alloc((2, 4, 64), "float32", strides=(8192, 64, 1), align=16)
    pool.commit()
    if thread < 64:
        scratch[0, 0, thread] = T.cast(thread + 7, "float32")
    T.cuda.cta_sync()
    if thread < 64:
        output[thread] = scratch[0, 0, thread]


@T.prim_func
def explicit_shared_strides_oob():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.SMEMPool()
    scratch = pool.alloc((2, 4, 64), "float32", strides=(8192, 64, 1), align=16)
    pool.commit()
    if lane == 0:
        scratch[1, 0, 0] = T.float32(1)


@T.prim_func
def dynamic_shared_tile_view_roundtrip(
    source: T.Buffer((64,), "int32"), output: T.Buffer((4, 4, 4), "int32")
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([LANES])
    shared = T.alloc_buffer((64,), "int32", scope="shared")
    if lane < 16:
        shared[warp * 16 + lane] = source[warp * 16 + lane]
    T.cuda.cta_sync()
    warp_view = shared.tile(0, (-1, 4, 4))[:, warp, :]
    if lane == 0:
        for outer in T.serial(4):
            for inner in T.serial(4):
                output[warp, outer, inner] = warp_view[outer * 4 + inner]


@T.prim_func
def tma_reduce_two_ctas(output: T.Buffer((4,), "float32")):
    T.device_entry()
    block = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([LANES])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(block + 1, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        Tx.copy_async(
            output[:],
            shared[:],
            dispatch="tma",
            use_tma_reduce="add",
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def tma_plain_store_two_ctas(output: T.Buffer((4,), "float32")):
    T.device_entry()
    block = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([LANES])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(block + 1, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        Tx.copy_async(output[:], shared[:], dispatch="tma")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def singleton_outer_gemm_b(
    a: T.Buffer((64, 128), "bfloat16"),
    b: T.Buffer((128, 128), "bfloat16"),
    sink: T.Buffer((128,), "int32"),
):
    T.device_entry()
    thread = T.thread_id([128])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([LANES])

    pool = T.SMEMPool()
    ready = TMABar(pool, 1)
    done = TCGen05Bar(pool, 1)
    ready.init(1)
    done.init(1)
    tmem_pool = T.TMEMPool(pool, total_cols=128, cta_group=1)

    pool.move_base_to(1024)
    a_smem = pool.alloc(
        (64, 128),
        "bfloat16",
        layout=mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (64, 128)),
        align=1024,
    )
    b_smem = pool.alloc(
        (1, 128, 128),
        "bfloat16",
        layout=mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, (1, 128, 128)),
        align=1024,
    )
    pool.commit()
    accumulator = tmem_pool.alloc(
        (64, 128),
        "float32",
        layout=tmem_datapath_layout("F", 64, 128),
    )
    tmem_pool.commit()

    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        if T.cuda.elect_sync():
            Tx.copy_async(a_smem, a, dispatch="tma", mbar=ready.ptr_to([0]))
            Tx.copy_async(b_smem[0], b, dispatch="tma", mbar=ready.ptr_to([0]))
            ready.arrive(0, tx_count=(64 * 128 + 128 * 128) * 2)
    ready.wait(0, 0)

    if warp == 0:
        Tx.warp.gemm_async(
            accumulator,
            a_smem,
            b_smem[0],
            transB=True,
            dispatch="tcgen05",
            cta_group=1,
        )
        if T.cuda.elect_sync():
            done.arrive(0, cta_group=1)
    done.wait(0, 0)
    T.ptx.tcgen05.fence__after_thread_sync()
    sink[thread] = warp + lane
    tmem_pool.dealloc()


@T.prim_func
def nested_elect_warp_gemm(
    a: T.Buffer((64, 128), "bfloat16"),
    b: T.Buffer((128, 128), "bfloat16"),
):
    T.device_entry()
    warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([LANES])

    pool = T.SMEMPool()
    tmem_pool = T.TMEMPool(pool, total_cols=128, cta_group=1)
    pool.move_base_to(1024)
    a_smem = pool.alloc(
        (64, 128),
        "bfloat16",
        layout=mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (64, 128)),
        align=1024,
    )
    b_smem = pool.alloc(
        (1, 128, 128),
        "bfloat16",
        layout=mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, (1, 128, 128)),
        align=1024,
    )
    pool.commit()
    accumulator = tmem_pool.alloc(
        (64, 128),
        "float32",
        layout=tmem_datapath_layout("F", 64, 128),
    )
    tmem_pool.commit()

    if warp == 0:
        if T.cuda.elect_sync():
            Tx.warp.gemm_async(
                accumulator,
                a_smem,
                b_smem[0],
                transB=True,
                dispatch="tcgen05",
                cta_group=1,
            )


@T.prim_func
def nested_elect_only(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    if T.cuda.elect_sync():
        if T.cuda.elect_sync():
            output[0] = 1


@T.prim_func
def elected_warp_register_copy(output: T.Buffer((LANES,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([LANES])
    source: T.f32[2]
    destination: T.f32[2]
    source[0] = T.cast(lane, "float32")
    source[1] = T.cast(lane + 1, "float32")
    if T.cuda.elect_sync():
        Tx.warp.copy(destination, source)
        output[lane] = destination[0] + destination[1]


@T.prim_func
def partial_warpgroup_register_op():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([LANES])
    tile = T.alloc_buffer((128, 1), "float32", scope="local", layout=wg_local_layout(1))
    if warp == 0:
        Tx.wg.fill(tile[:, :], T.float32(1))


@T.prim_func
def partial_lane_warpgroup_register_op():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([LANES])
    tile = T.alloc_buffer((128, 1), "float32", scope="local", layout=wg_local_layout(1))
    if warp == 0 and lane < LANES // 2:
        Tx.wg.fill(tile[:, :], T.float32(1))


@T.prim_func
def elected_tcgen_wait_ld():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    if T.cuda.elect_sync():
        T.ptx.tcgen05.wait__ld.sync.aligned()


@T.prim_func
def elected_tcgen_wait_st():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    if T.cuda.elect_sync():
        T.ptx.tcgen05.wait__st.sync.aligned()


@T.prim_func
def tma_padded_narrow_rows(source: T.Buffer((64, 8), "bfloat16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    pool = T.SMEMPool()
    ready = TMABar(pool, 1)
    ready.init(1)
    pool.move_base_to(1024)
    shared = pool.alloc(
        (64, 16),
        "bfloat16",
        layout=mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, (64, 16)),
        align=1024,
    )
    pool.commit()

    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if T.cuda.elect_sync():
        Tx.copy_async(
            shared[:, 0:8],
            source[:, :],
            dispatch="tma_auto",
            mbar=ready.ptr_to([0]),
        )
        ready.arrive(0, tx_count=64 * 8 * 2)
    ready.wait(0, 0)


@T.prim_func
def tma_aligned_full_rows(source: T.Buffer((64, 16), "bfloat16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([LANES])
    pool = T.SMEMPool()
    ready = TMABar(pool, 1)
    ready.init(1)
    pool.move_base_to(1024)
    shared = pool.alloc(
        (64, 16),
        "bfloat16",
        layout=mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, (64, 16)),
        align=1024,
    )
    pool.commit()

    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if T.cuda.elect_sync():
        Tx.copy_async(
            shared[:, :],
            source[:, :],
            dispatch="tma",
            mbar=ready.ptr_to([0]),
        )
        ready.arrive(0, tx_count=64 * 16 * 2)
    ready.wait(0, 0)


def _bfloat16_zeros(
    shape: tuple[int, ...],
):
    return np.zeros(shape, dtype=np.uint16)


def test_lane_owner_layout_has_one_private_register_slot_per_lane(tmp_path):
    source = np.arange(LANES, dtype=np.float32)
    module = numsim.transpile(lane_owner_warp_sum, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": np.zeros_like(source),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.full(LANES, source.sum(dtype=np.float32), dtype=np.float32),
    )


def test_one_register_m64_fragment_roundtrips_all_thread_values(tmp_path):
    module = numsim.transpile(one_register_m64_fragment, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((128,), dtype=np.float32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.arange(128, dtype=np.float32),
    )


def test_explicit_shared_strides_use_executed_affine_addresses(tmp_path):
    module = numsim.transpile(explicit_shared_strides_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((64,), dtype=np.float32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.arange(64, dtype=np.float32) + np.float32(7),
    )


def test_dynamic_shared_tile_view_uses_its_static_parent_bounds(tmp_path):
    source = np.arange(64, dtype=np.int32)
    output = np.full((4, 4, 4), -1, dtype=np.int32)

    module = numsim.transpile(dynamic_shared_tile_view_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = source.reshape(4, 4, 4).transpose(1, 0, 2)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    "checker",
    [
        pytest.param(synccheck, id="synccheck"),
        pytest.param(racecheck, id="racecheck"),
    ],
)
@pytest.mark.parametrize(
    ("kernel", "inputs"),
    [
        pytest.param(
            lane_owner_warp_sum,
            {
                "source": np.arange(LANES, dtype=np.float32),
                "output": np.zeros((LANES,), dtype=np.float32),
            },
            id="lane-owner-layout",
        ),
        pytest.param(
            one_register_m64_fragment,
            {"output": np.zeros((128,), dtype=np.float32)},
            id="one-register-m64-fragment",
        ),
        pytest.param(
            explicit_shared_strides_roundtrip,
            {"output": np.zeros((64,), dtype=np.float32)},
            id="explicit-shared-strides",
        ),
    ],
)
def test_reported_layouts_reach_each_semantic_checker(checker, kernel, inputs):
    checker(kernel, inputs=inputs).require_clean()


