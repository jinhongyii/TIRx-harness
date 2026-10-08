"""v2 port of the legacy per-operation Synccheck protocol runtime test.

Legacy: tests/analysis_tools/synccheck/runtime/test_device_protocol_ops.py::test_protocol_runtime
analyzed and transpiled all kernels below into one _analysis_capable module and
ran Engine.run_synccheck_phase per kernel. For every op _KERNEL_BY_OP assigns
to the kernel it asserted the op is present in the legacy ``analyze`` source
map, the clean verdict contract, and (via the legacy ``effects`` trace keyed by
source op id) that the op produced the _EFFECTS_BY_OP protocol effects.

The v2 copy runs v2.synccheck on each kernel with the legacy coverage bounds
and resource limits and keeps: phase name, no execution error, no findings,
verdict clean, no incomplete, coverage.eligible_for_clean and, for kernels
owning cp.async / bulk-async group ops, a non-zero completion count (legacy
stats["completion_operation_count"] > 0, v2 stats["completions"] > 0).
Dropped: the source-op-id presence check (legacy ``analyze`` source map), the
per-op ``effects`` assertion (v2 phase payloads carry no effect trace),
stats["task_count"] / stats["completed_task_count"], search["algorithm"],
the legacy-only engine knobs (max_workers, native_loop_*, max_polls,
max_transitions) and _analysis_capable. Kernels are copied verbatim from
their legacy modules (each block is marked with its source path).
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.backend.cuda.lang.clc import query_cancel_first_ctaid_x
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


# --- copied from tests/numsim/runtime/test_sync_runtime_domain_oracle.py ---

@T.prim_func
def clc_no_work_cluster_acquire_wait(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    first_ctaid_x = T.local_scalar("uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), T.uint32(16))
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](T.address_of(response[0]), T.address_of(barrier[0]))
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barrier[0]), 0)
        query_cancel_first_ctaid_x(first_ctaid_x, T.address_of(response[0]))
        output[0] = first_ctaid_x


# --- copied from tests/analysis_tools/synccheck/runtime/test_device_protocol_ops.py ---

@T.prim_func
def collective_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([4])
    _lane = T.lane_id([32])

    T.cuda.warp_sync()
    T.cuda.warpgroup_sync(7)
    T.cuda.cta_sync()
    T.cuda.cluster_sync()
    T.cuda.grid_sync()


@T.prim_func
def named_barrier_protocol_ops():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.bar.arrive(T.uint32(3), T.uint32(64))
    else:
        T.ptx.bar.sync(T.uint32(3), T.uint32(64))
    if warp == 0:
        T.ptx.bar.arrive(T.uint32(4), T.uint32(64))
    else:
        T.ptx.barrier.sync(T.uint32(4), T.uint32(64))
    T.ptx.bar.sync(T.uint32(5))
    T.ptx.barrier.sync(T.uint32(6))
    if warp == 0:
        T.ptx.barrier.arrive(T.uint32(7), T.uint32(64))
    else:
        T.ptx.barrier.sync(T.uint32(7), T.uint32(64))
    T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))


@T.prim_func
def split_cluster_barrier_protocol_ops():
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])

    T.ptx.barrier.cluster.arrive()
    T.ptx.barrier.cluster.wait()


@T.prim_func
def mbarrier_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((6,), "uint64", scope="shared", align=8)
    state = T.local_scalar("uint64")

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[2]), 2)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[3]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[4]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[5]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[4]), T.uint32(1))
        T.cuda.mbarrier_wait(T.address_of(barriers[4]), 0)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[1]), 0)
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barriers[1]), 0)
        T.ptx.mbarrier.arrive.noComplete.shared.b64(state, T.address_of(barriers[2]), T.uint32(1))
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[2]))
        T.cuda.mbarrier_wait(T.address_of(barriers[2]), 0)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[3]), 16)
        T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(
            T.address_of(barriers[3]),
            T.uint32(16),
            pred=T.uint32(1),
        )
        T.cuda.mbarrier_wait(T.address_of(barriers[3]), 0)
        T.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
            T.address_of(barriers[5]), T.uint32(16)
        )
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[5]))
        T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(
            T.address_of(barriers[5]), T.uint32(16)
        )
        T.cuda.mbarrier_wait(T.address_of(barriers[5]), 0)


@T.prim_func
def clc_protocol_op():
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](T.address_of(response[0]), T.address_of(barrier[0]))
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barrier[0]), 0)


@T.prim_func
def ordering_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])

    T.cuda.thread_fence()
    T.cuda.nano_sleep(0)
    T.cuda.printf("native synccheck protocol runtime %d", 7)
    T.ptx.fence.sc.cta()
    T.ptx.griddepcontrol.launch_dependents()
    T.ptx.griddepcontrol.wait()
    T.cuda.warp_sync()


@T.prim_func
def setmaxnreg_protocol_op():
    T.device_entry()
    _wg = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])

    T.ptx.setmaxnreg.dec.sync.aligned.u32(88)


@T.prim_func
def tcgen_lifecycle_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared", align=4)

    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tcgen_commit_protocol_op():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)


@T.prim_func
def tcgen_ordering_protocol_ops(output: T.Buffer((4, 32), "uint32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    value = T.alloc_local((1,), "uint32")

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.cta_sync()

    value[0] = T.cast(1000 + warp * 32 + lane, "uint32")
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    value[0] = T.uint32(0)
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[warp, lane] = value[0]
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def cp_async_group_protocol_ops(
    source: T.Buffer((128,), "uint8"), output: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")

    T.ptx["cp.async.ca.shared.global.L2::64B"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        4,
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.prim_func
def cp_async_zero_fill_group_protocol_ops(source: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")

    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        4,
        T.cast(T.if_then_else(lane < 16, 4, 0), "uint32"),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)


@T.prim_func
def bulk_async_group_protocol_ops(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")

    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            output.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32")
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
    T.cuda.warp_sync()


@T.prim_func
def pure_warp_sync_source_ops(output: T.Buffer((32, 10), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full: T.let = T.uint32(0xFFFFFFFF)
    value: T.let = T.cast(lane + 1, "uint32")

    output[lane, 0] = T.cuda.__shfl_sync(full, value, T.cast(31 - lane, "uint32"), 32)
    output[lane, 1] = T.cuda.__shfl_up_sync(full, value, 1, 32)
    output[lane, 2] = T.cuda.__shfl_down_sync(full, value, 1, 32)
    output[lane, 3] = T.cuda.__shfl_xor_sync(full, value, 1, 32)
    output[lane, 4] = T.cuda.ballot_sync(full, lane < 16)
    output[lane, 5] = T.cuda.reduce_add_sync_u32(full, value)
    output[lane, 6] = T.cuda.reduce_min_sync_u32(full, value)
    output[lane, 7] = T.cast(T.cuda.any_sync(full, lane == 31), "uint32")
    output[lane, 8] = T.cuda.warp_sum(value, width=8)
    output[lane, 9] = T.cuda.elect_sync()


@T.prim_func
def pure_cta_sync_source_ops(output: T.Buffer((2, 32, 3), "int64")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((2,), "int32", scope="shared")
    value: T.let = warp * 32 + lane + 1

    output[warp, lane, 0] = T.cast(T.cuda.cta_sum(value, 2, scratch.ptr_to([0])), "int64")
    output[warp, lane, 1] = T.cuda.syncthreads_and(value > 0)
    output[warp, lane, 2] = T.cuda.syncthreads_or(lane == 31)


@T.prim_func
def mbarrier_query_sync_source_ops(output: T.Buffer((32, 4), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.ptx.mbarrier.test_wait.parity.shared.b64(
        output[lane, 0], T.address_of(barrier[0]), T.uint32(1)
    )
    T.ptx.mbarrier.try_wait.parity.shared.b64(
        output[lane, 1], T.address_of(barrier[0]), T.uint32(0), T.uint32(1)
    )
    T.ptx.mbarrier.try_wait.parity.shared.b64(
        output[lane, 2], T.address_of(barrier[0]), T.uint32(0)
    )
    if lane == 0:
        token = T.alloc_buffer((1,), "uint64", scope="local")
        barrier_address: T.let = T.cuda.cvta_generic_to_shared(T.address_of(barrier[0]))
        T.ptx.mbarrier.arrive.shared__cta.b64(token[0], barrier_address, T.uint32(1))
        T.ptx.mbarrier.try_wait.shared__cta.b64(output[lane, 3], barrier_address, token[0])


_KERNEL_BY_OP = {
    "tirx.cuda.__shfl_down_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.__shfl_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.__shfl_up_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.__shfl_xor_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.ballot_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.cluster_sync": "collective_protocol_ops",
    "tirx.cuda.cta_reduce": "pure_cta_sync_source_ops",
    "tirx.cuda.cta_sync": "collective_protocol_ops",
    "tirx.cuda.grid_sync": "collective_protocol_ops",
    "tirx.cuda.nano_sleep": "ordering_protocol_ops",
    "tirx.cuda.printf": "ordering_protocol_ops",
    "tirx.cuda.reduce_add_sync_u32": "pure_warp_sync_source_ops",
    "tirx.cuda.reduce_min_sync_u32": "pure_warp_sync_source_ops",
    "tirx.cuda.syncthreads_and": "pure_cta_sync_source_ops",
    "tirx.cuda.syncthreads_or": "pure_cta_sync_source_ops",
    "tirx.cuda.thread_fence": "ordering_protocol_ops",
    "tirx.cuda.warp_reduce": "pure_warp_sync_source_ops",
    "tirx.cuda.warp_sync": "collective_protocol_ops",
    "tirx.cuda.warpgroup_sync": "collective_protocol_ops",
    "tirx.ptx.bar_arrive": "named_barrier_protocol_ops",
    "tirx.ptx.bar_sync": "named_barrier_protocol_ops",
    "tirx.ptx.bar_sync_count": "named_barrier_protocol_ops",
    "tirx.ptx.bar_warp_sync": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_arrive": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_sync": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_sync_count": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_cluster_arrive": "split_cluster_barrier_protocol_ops",
    "tirx.ptx.barrier_cluster_wait": "split_cluster_barrier_protocol_ops",
    "tirx.cuda.any_sync": "pure_warp_sync_source_ops",
    "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid": (
        "clc_no_work_cluster_acquire_wait"
    ),
    "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled": ("clc_no_work_cluster_acquire_wait"),
    "tirx.ptx.clusterlaunchcontrol_try_cancel": "clc_protocol_op",
    "tirx.ptx.cp_async_bulk_commit_group": "bulk_async_group_protocol_ops",
    "tirx.ptx.cp_async_bulk_wait_group": "bulk_async_group_protocol_ops",
    "tirx.ptx.cp_async_ca": "cp_async_group_protocol_ops",
    "tirx.ptx.cp_async_ca_src_size": "cp_async_zero_fill_group_protocol_ops",
    "tirx.ptx.cp_async_commit_group": "cp_async_group_protocol_ops",
    "tirx.ptx.cp_async_wait_group": "cp_async_group_protocol_ops",
    "tirx.cuda.elect_sync": "pure_warp_sync_source_ops",
    "tirx.ptx.fence": "ordering_protocol_ops",
    "tirx.ptx.fence_mbarrier_init": "mbarrier_protocol_ops",
    "tirx.ptx.fence_proxy": "mbarrier_protocol_ops",
    "tirx.ptx.griddepcontrol": "ordering_protocol_ops",
    "tirx.ptx.mbarrier_arrive": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_arrive_count_state": "mbarrier_query_sync_source_ops",
    "tirx.ptx.mbarrier_arrive_nocount": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_arrive_no_complete": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_arrive_expect_tx": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_expect_tx": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_complete_tx": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_init": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_test_wait_parity": "mbarrier_query_sync_source_ops",
    "tirx.cuda.mbarrier_wait": "mbarrier_protocol_ops",
    "tirx.cuda.mbarrier_wait_acquire_cluster": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_try_wait_parity": "mbarrier_query_sync_source_ops",
    "tirx.ptx.mbarrier_try_wait_parity_no_hint": "mbarrier_query_sync_source_ops",
    "tirx.ptx.mbarrier_try_wait": "mbarrier_query_sync_source_ops",
    "tirx.ptx.setmaxnreg": "setmaxnreg_protocol_op",
    "tirx.ptx.tcgen05_alloc": "tcgen_lifecycle_protocol_ops",
    "tirx.ptx.tcgen05_commit": "tcgen_commit_protocol_op",
    "tirx.ptx.tcgen05_dealloc": "tcgen_lifecycle_protocol_ops",
    "tirx.ptx.tcgen05_fence": "tcgen_ordering_protocol_ops",
    "tirx.ptx.tcgen05_ld": "tcgen_ordering_protocol_ops",
    "tirx.ptx.tcgen05_relinquish_alloc_permit": "tcgen_lifecycle_protocol_ops",
    "tirx.ptx.tcgen05_wait": "tcgen_ordering_protocol_ops",
    "tirx.ptx.tcgen05_st": "tcgen_ordering_protocol_ops",
}


_CP_ASYNC_COMPLETION_OPS = frozenset(
    {
        "tirx.ptx.cp_async_ca",
        "tirx.ptx.cp_async_ca_src_size",
        "tirx.ptx.cp_async_commit_group",
        "tirx.ptx.cp_async_wait_group",
    }
)


_BULK_ASYNC_GROUP_OPS = frozenset(
    {
        "tirx.ptx.cp_async_bulk_commit_group",
        "tirx.ptx.cp_async_bulk_wait_group",
    }
)


_KERNELS = (
    collective_protocol_ops,
    named_barrier_protocol_ops,
    split_cluster_barrier_protocol_ops,
    mbarrier_protocol_ops,
    clc_protocol_op,
    clc_no_work_cluster_acquire_wait,
    ordering_protocol_ops,
    setmaxnreg_protocol_op,
    tcgen_lifecycle_protocol_ops,
    tcgen_commit_protocol_op,
    tcgen_ordering_protocol_ops,
    cp_async_group_protocol_ops,
    cp_async_zero_fill_group_protocol_ops,
    bulk_async_group_protocol_ops,
    pure_warp_sync_source_ops,
    pure_cta_sync_source_ops,
    mbarrier_query_sync_source_ops,
)

_BY_NAME = {kernel.__name__: kernel for kernel in _KERNELS}


def _inputs(kernel_name: str) -> dict[str, np.ndarray]:
    """The legacy NativeProtocolArtifact.inputs() bindings of one kernel."""

    bindings: dict[str, np.ndarray] = {}
    if kernel_name == "tcgen_ordering_protocol_ops":
        bindings["output"] = np.zeros((4, 32), dtype=np.uint32)
    for name, size in (
        ("cp_async_group_protocol_ops", 128),
        ("cp_async_zero_fill_group_protocol_ops", 128),
        ("bulk_async_group_protocol_ops", 16),
    ):
        if name == kernel_name:
            bindings["source"] = np.arange(size, dtype=np.uint8)
            if name in {"cp_async_group_protocol_ops", "bulk_async_group_protocol_ops"}:
                bindings["output"] = np.zeros(size, dtype=np.uint8)
    for name, shape, dtype in (
        ("clc_no_work_cluster_acquire_wait", (1,), np.uint32),
        ("pure_warp_sync_source_ops", (32, 10), np.uint32),
        ("pure_cta_sync_source_ops", (2, 32, 3), np.int64),
        ("mbarrier_query_sync_source_ops", (32, 4), np.uint32),
    ):
        if name == kernel_name:
            bindings["output"] = np.zeros(shape, dtype=dtype)
    return bindings


@pytest.mark.parametrize("kernel_name", sorted(set(_KERNEL_BY_OP.values())))
def test_protocol_runtime(kernel_name):
    """Port of tests/analysis_tools/synccheck/runtime/test_device_protocol_ops.py::test_protocol_runtime.

    Dropped: source-op-id presence, per-op ``effects``, task counts and
    search["algorithm"] (see the module docstring).
    """

    assert kernel_name in _BY_NAME
    report = v2.synccheck(
        _BY_NAME[kernel_name],
        _inputs(kernel_name),
        coverage_bounds=v2.CoverageBounds(0, 0),
        resource_limits=v2.ResourceLimits(
            max_schedules=100,
            max_backtrack_nodes=10_000,
            max_events_per_run=10_000,
            max_total_events=100_000,
            max_loop_steps=100_000,
            max_wall_time_ms=30_000,
            max_diagnostic_bytes=1_000_000,
        ),
    )
    assert len(report.phases) == 1
    result = report.phases[0].to_dict()
    assert result["phase"]["name"] == kernel_name
    assert result["execution_error"] is None, report.format()
    assert result["findings"] == [], report.format()
    assert result["verdict"] == "clean", report.format()
    assert result["incomplete"] == [], report.format()
    assert result["coverage"]["eligible_for_clean"] is True
    owned = {op for op, owner in _KERNEL_BY_OP.items() if owner == kernel_name}
    if owned & (_CP_ASYNC_COMPLETION_OPS | _BULK_ASYNC_GROUP_OPS):
        assert result["stats"]["completions"] > 0, result["stats"]
