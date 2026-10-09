from __future__ import annotations

import numpy as np
import pytest
import tvm

from tirx_harness import numsim, racecheck, synccheck
from tvm.script import tirx as T


@T.prim_func
def lane_varying_blocking_wait(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if lane < 2:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if lane < 2:
        T.cuda.mbarrier_wait(barriers.ptr_to([lane]), 1 - lane)
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([1]))
        output[lane] = lane + 7


@T.prim_func
def checker_lane_varying_pending_wait(output: T.Buffer((4,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((4,), "uint64", scope="shared")

    if warp == 0 and lane < 4:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 1:
        if lane < 4:
            T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([lane]))
            T.cuda.mbarrier_wait(barriers.ptr_to([lane]), 0)
    T.cuda.cta_sync()

    # Keep four distinct barrier requests pending in one warp instruction until
    # the producer warp runs. The old checker rejected this legal lane grouping.
    if warp == 1:
        if lane < 4:
            T.cuda.mbarrier_wait(barriers.ptr_to([lane]), 1)
            output[lane] = lane + 1
    else:
        if lane < 4:
            T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([lane]))


@T.prim_func
def lane_varying_expect_tx(source: T.Buffer((64,), "uint8"), output: T.Buffer((64,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint8", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    shared[lane] = T.uint8(0)
    shared[lane + 32] = T.uint8(0)
    if lane < 2:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), source.ptr_to([0]), 16, barriers.ptr_to([0])
        )
    if lane == 1:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([1]), 16)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([32]), source.ptr_to([32]), 16, barriers.ptr_to([1])
        )
    if lane < 2:
        T.cuda.mbarrier_wait(barriers.ptr_to([lane]), 0)
    T.cuda.warp_sync()

    output[lane] = shared[lane]
    output[lane + 32] = shared[lane + 32]


@T.prim_func
def lane_varying_nonblocking_queries(output: T.Buffer((2, 4), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if lane < 2:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([0]))
    T.cuda.warp_sync()

    if lane < 2:
        T.ptx.mbarrier.test_wait.parity.shared.b64(
            output[lane, 0], barriers.ptr_to([lane]), T.cast(lane, "uint32")
        )
        T.ptx.mbarrier.try_wait.parity.shared.b64(
            output[lane, 1],
            barriers.ptr_to([lane]),
            T.cast(lane, "uint32"),
            T.uint32(lane + 1),
        )
        T.ptx["mbarrier.test_wait.parity.phase_type::conditional.shared.b64"](
            output[lane, 2], barriers.ptr_to([lane]), T.cast(lane, "uint32")
        )
        T.ptx["mbarrier.try_wait.parity.phase_type::conditional.shared.b64"](
            output[lane, 3], barriers.ptr_to([lane]), T.cast(lane, "uint32"), T.uint32(lane + 1)
        )
    T.cuda.warp_sync()
    if lane == 1:
        T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([1]))
        T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)


@T.prim_func
def lane_varying_remote_arrive(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if lane < 2:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()

    if cta == 0:
        remote_barrier = T.alloc_local((1,), "uint64")
        T.ptx.mapa.shared__cluster.u64(
            remote_barrier[0], barriers.ptr_to([lane % 2]), T.uint32(lane % 2)
        )
        T.ptx.mbarrier.arrive.b64(remote_barrier[0], T.uint32(1), pred=lane < 2)
    if (cta == 0) and (lane == 0):
        T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        output[0] = 1
    if (cta == 1) and (lane == 1):
        T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)
        output[1] = 1
    T.cuda.cluster_sync()


@T.prim_func
def current_mbarrier_completion_forms(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared", align=8)
    state = T.local_scalar("uint64")

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 2)
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([1]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if lane == 0:
        T.ptx.mbarrier.arrive.noComplete.shared.b64(state, barriers.ptr_to([0]), T.uint32(1))
        T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([0]))
        T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        output[0] = 1

        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([1]), T.uint32(16))
        T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(barriers.ptr_to([1]), T.uint32(16))
        T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)
        output[1] = 1


@T.prim_func
def standalone_mbarrier_expect_tx(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if lane == 0:
        T.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(barrier.ptr_to([0]), T.uint32(16))
        # A standalone expectation must not also consume this arrival.
        T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]))
        T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(barrier.ptr_to([0]), T.uint32(16))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        output[0] = 1


@T.prim_func
def elected_mbarrier_init(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    leader = T.cuda.elect_sync()
    T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1, pred=leader)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        output[0] = 1


def test_issue_time_numeric_completions_allow_distinct_barrier_waits(tmp_path):
    source = np.arange(1, 65, dtype=np.uint8)
    module = numsim.transpile(lane_varying_expect_tx, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.full(64, np.uint8(0xEE), dtype=np.uint8)}
    )

    expected = np.zeros(64, dtype=np.uint8)
    expected[:16] = source[:16]
    expected[32:48] = source[32:48]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_nonblocking_queries_snapshot_lane_varying_conditions(tmp_path):
    module = numsim.transpile(lane_varying_nonblocking_queries, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((2, 4), dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.ones((2, 4), dtype=np.uint32))


@pytest.mark.parametrize("checker", [racecheck, synccheck])
def test_checkers_allow_lane_varying_nonblocking_queries(checker):
    report = checker(
        lane_varying_nonblocking_queries,
        {"output": np.zeros((2, 4), dtype=np.uint32)},
    )

    assert report.verdict == "clean", report.format()


def test_remote_arrive_honors_lane_varying_target_and_predicate(tmp_path):
    module = numsim.transpile(lane_varying_remote_arrive, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_current_mbarrier_arrival_and_completion_forms_advance_protocol(tmp_path):
    module = numsim.transpile(current_mbarrier_completion_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_standalone_mbarrier_expect_tx_arms_without_consuming_an_arrival(tmp_path):
    module = numsim.transpile(standalone_mbarrier_expect_tx, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.int32))


def test_mbarrier_init_honors_elected_lane_predicate(tmp_path):
    module = numsim.transpile(elected_mbarrier_init, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.int32))


