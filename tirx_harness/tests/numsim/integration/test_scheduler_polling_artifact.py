from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def scheduler_progress_poll(
    root_head: T.Buffer((1,), "int32"),
    root_total: T.Buffer((1,), "int32"),
    root_ready_queue: T.Buffer((1,), "int32"),
    cont_head: T.Buffer((1,), "int32"),
    cont_tail: T.Buffer((1,), "int32"),
    cont_ready_queue: T.Buffer((1,), "int32"),
    done_counter: T.Buffer((1,), "int32"),
    work_total: T.Buffer((1,), "int32"),
    publish_done: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    ready_work = T.local_scalar("int32")
    atomic_old = T.local_scalar("int32")
    root_seen = T.local_scalar("int32")
    cont_tail_seen = T.local_scalar("int32")
    cont_head_seen = T.local_scalar("int32")
    done_seen = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            ready_work = T.int32(-1)
            while ready_work < T.int32(0):
                T.ptx.ld.acquire.gpu.global_.s32(root_seen, root_head.ptr_to([0]))
                if root_seen < root_total[0]:
                    root_pos: T.let = T.cuda.atomic_add(root_head.ptr_to([0]), T.int32(1))
                    if root_pos < root_total[0]:
                        ready_work = root_ready_queue[root_pos]
                else:
                    T.ptx.ld.acquire.gpu.global_.s32(cont_tail_seen, cont_tail.ptr_to([0]))
                    T.ptx.ld.acquire.gpu.global_.s32(cont_head_seen, cont_head.ptr_to([0]))
                    if cont_head_seen < cont_tail_seen:
                        cont_old: T.let = T.cuda.atomic_cas(
                            cont_head.ptr_to([0]), cont_head_seen, cont_head_seen + T.int32(1)
                        )
                        if cont_old == cont_head_seen:
                            ready_work = cont_ready_queue[cont_head_seen]
                    else:
                        T.ptx.ld.acquire.gpu.global_.s32(done_seen, done_counter.ptr_to([0]))
                        if done_seen >= work_total[0]:
                            ready_work = work_total[0]
                        else:
                            T.evaluate(0)
            output[0] = ready_work
    elif lane == 0:
        if publish_done[0] != T.int32(0):
            T.ptx.atom.release.gpu.global_.add.s32(atomic_old, done_counter.ptr_to([0]), T.int32(1))
        else:
            cont_ready_queue[0] = T.int32(7)
            T.ptx.atom.release.gpu.global_.add.s32(atomic_old, cont_tail.ptr_to([0]), T.int32(1))


@T.prim_func
def scheduler_progress_poll_two_lane_cas_mismatch(
    root_head: T.Buffer((1,), "int32"),
    root_total: T.Buffer((1,), "int32"),
    root_ready_queue: T.Buffer((1,), "int32"),
    cont_head: T.Buffer((1,), "int32"),
    cont_tail: T.Buffer((1,), "int32"),
    cont_ready_queue: T.Buffer((1,), "int32"),
    done_counter: T.Buffer((1,), "int32"),
    work_total: T.Buffer((1,), "int32"),
    output: T.Buffer((2,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    ready_work = T.local_scalar("int32")
    root_seen = T.local_scalar("int32")
    cont_tail_seen = T.local_scalar("int32")
    cont_head_seen = T.local_scalar("int32")
    done_seen = T.local_scalar("int32")
    if lane < 2:
        ready_work = T.int32(-1)
        while ready_work < T.int32(0):
            T.ptx.ld.acquire.gpu.global_.s32(root_seen, root_head.ptr_to([0]))
            if root_seen < root_total[0]:
                root_pos: T.let = T.cuda.atomic_add(root_head.ptr_to([0]), T.int32(1))
                if root_pos < root_total[0]:
                    ready_work = root_ready_queue[root_pos]
            else:
                T.ptx.ld.acquire.gpu.global_.s32(cont_tail_seen, cont_tail.ptr_to([0]))
                T.ptx.ld.acquire.gpu.global_.s32(cont_head_seen, cont_head.ptr_to([0]))
                if cont_head_seen < cont_tail_seen:
                    cont_old: T.let = T.cuda.atomic_cas(
                        cont_head.ptr_to([0]), cont_head_seen, cont_head_seen + T.int32(1)
                    )
                    if cont_old == cont_head_seen:
                        ready_work = cont_ready_queue[cont_head_seen]
                else:
                    T.ptx.ld.acquire.gpu.global_.s32(done_seen, done_counter.ptr_to([0]))
                    if done_seen >= work_total[0]:
                        ready_work = work_total[0]
                    else:
                        T.evaluate(0)
        output[lane] = ready_work


@T.prim_func
def scheduler_progress_poll_effectful_fallback(
    root_head: T.Buffer((1,), "int32"),
    root_total: T.Buffer((1,), "int32"),
    root_ready_queue: T.Buffer((1,), "int32"),
    cont_head: T.Buffer((1,), "int32"),
    cont_tail: T.Buffer((1,), "int32"),
    cont_ready_queue: T.Buffer((1,), "int32"),
    done_counter: T.Buffer((1,), "int32"),
    work_total: T.Buffer((1,), "int32"),
    output: T.Buffer((2,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    ready_work = T.local_scalar("int32")
    root_seen = T.local_scalar("int32")
    cont_tail_seen = T.local_scalar("int32")
    cont_head_seen = T.local_scalar("int32")
    done_seen = T.local_scalar("int32")
    if lane == 0:
        ready_work = T.int32(-1)
        while ready_work < T.int32(0):
            T.ptx.ld.acquire.gpu.global_.s32(root_seen, root_head.ptr_to([0]))
            if root_seen < root_total[0]:
                root_pos: T.let = T.cuda.atomic_add(root_head.ptr_to([0]), T.int32(1))
                if root_pos < root_total[0]:
                    ready_work = root_ready_queue[root_pos]
            else:
                T.ptx.ld.acquire.gpu.global_.s32(cont_tail_seen, cont_tail.ptr_to([0]))
                T.ptx.ld.acquire.gpu.global_.s32(cont_head_seen, cont_head.ptr_to([0]))
                if cont_head_seen < cont_tail_seen:
                    cont_old: T.let = T.cuda.atomic_cas(
                        cont_head.ptr_to([0]), cont_head_seen, cont_head_seen + T.int32(1)
                    )
                    if cont_old == cont_head_seen:
                        ready_work = cont_ready_queue[cont_head_seen]
                else:
                    T.ptx.ld.acquire.gpu.global_.s32(done_seen, done_counter.ptr_to([0]))
                    if done_seen >= work_total[0]:
                        ready_work = work_total[0]
                    else:
                        output[1] = output[1] + T.int32(1)
        output[0] = ready_work


@T.prim_func
def time_sliced_side_effect_poll(signal: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    current = T.local_scalar("int32")
    atomic_old = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            while T.int32(1):
                T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([0]))
                if current != T.int32(0):
                    break
                output[0] = output[0] + T.int32(1)
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.s32(atomic_old, signal.ptr_to([0]), T.int32(1))


@T.prim_func
def time_sliced_finite_for(signal: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    current = T.local_scalar("int32")
    atomic_old = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            for _step in T.serial(128):
                T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([0]))
                if current == T.int32(0):
                    output[0] = output[0] + T.int32(1)
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.s32(atomic_old, signal.ptr_to([0]), T.int32(1))


@T.prim_func
def time_sliced_lane_divergent_while(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    count = T.local_scalar("int32")
    count = T.int32(0)
    while count < lane + T.int32(64):
        count = count + T.int32(1)
    output[lane] = count


@T.prim_func
def short_native_while(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    count = T.local_scalar("int32")
    if lane == 0:
        count = T.int32(0)
        while count < T.int32(63):
            count = count + T.int32(1)
        output[0] = count


@T.prim_func
def long_native_while(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    count = T.local_scalar("int32")
    if lane == 0:
        count = T.int32(0)
        while count < T.int32(1_000_001):
            count = count + T.int32(1)
        output[0] = count


@T.prim_func
def full_continue_reaches_time_slice(output: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    count = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            count = T.int32(0)
            while count < T.int32(64):
                count = count + T.int32(1)
                continue
            output[0] = count
    elif lane == 0:
        output[1] = T.int32(1)


@T.prim_func
def initially_false_native_while(control: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        while control[0] < T.int32(0):
            output[0] = output[0] + T.int32(1)


@T.prim_func
def nested_time_sliced_while(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    outer = T.local_scalar("int32")
    inner = T.local_scalar("int32")
    total = T.local_scalar("int32")
    if lane == 0:
        outer = T.int32(0)
        total = T.int32(0)
        while outer < T.int32(65):
            inner = T.int32(0)
            while inner < T.int32(3):
                total = total + T.int32(1)
                break
            outer = outer + T.int32(1)
        output[0] = total


@T.prim_func
def integer_truth_while(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        while T.int32(1):
            output[0] = output[0] + T.int32(1)
            break
        while T.int32(0):
            output[1] = output[1] + T.int32(1)


def _scheduler_args(*, root_total: int = 0, publish_done: bool = False) -> dict[str, np.ndarray]:
    return {
        "root_head": np.zeros(1, dtype=np.int32),
        "root_total": np.asarray([root_total], dtype=np.int32),
        "root_ready_queue": np.asarray([5], dtype=np.int32),
        "cont_head": np.zeros(1, dtype=np.int32),
        "cont_tail": np.zeros(1, dtype=np.int32),
        "cont_ready_queue": np.zeros(1, dtype=np.int32),
        "done_counter": np.zeros(1, dtype=np.int32),
        "work_total": np.ones(1, dtype=np.int32),
        "publish_done": np.asarray([publish_done], dtype=np.int32),
        "output": np.zeros(1, dtype=np.int32),
    }


def test_scheduler_progress_loop_does_not_replay_successful_root_claim(tmp_path):
    module = numsim.transpile(scheduler_progress_poll, cache_dir=tmp_path)
    result = numsim.Engine().run(module, _scheduler_args(root_total=1))

    np.testing.assert_array_equal(result.outputs["output"], np.asarray([5], dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["root_head"], np.ones(1, dtype=np.int32))


def test_integer_while_conditions_use_nonzero_truth_semantics(tmp_path):
    module = numsim.transpile(integer_truth_while, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.asarray([1, 0], dtype=np.int32))
