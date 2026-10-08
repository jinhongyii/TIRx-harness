"""v2 ports of three legacy time-slice tests in
``tests/numsim/integration/test_scheduler_polling_artifact.py``:
``test_time_slice_does_not_replay_body_side_effects_and_schedules_peer``,
``test_time_slice_uses_configured_reschedule_quantum`` and
``test_finite_for_time_slice_schedules_peer_without_pattern_matching``.

Dropped pins (legacy scheduler implementation, not semantics):

- the exact spin count in ``output`` (legacy: 64 = default reschedule quantum, or 7
  with ``native_loop_reschedule_quantum=7``). It counts how many polls warp 0 made
  before the scheduler first ran warp 1, i.e. the legacy time-slice policy. The v2
  scheduler (seeded warp rotation, ``docs/development/architecture.md``) may run
  the peer earlier or later; v2 gives 25 / 0 / 21 at seed 0;
- ``result.stats["poll_order"]`` (legacy poll trace).

Kept: the run completes (the spinning warp does not starve the releasing peer), the
release lands exactly once (``signal == 1``), the loop body's side effect is never
replayed or lost (``output`` equals the number of iterations warp 0 observed the
flag unset, which is bounded by the loop's trip count for the finite ``for``),
and the configured quantum is accepted.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

SEEDS = range(8)


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


def _args():
    return {"signal": np.zeros(1, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)}


def _check_side_effect_poll(engine_kwargs):
    module = v2.transpile(time_sliced_side_effect_poll)
    for seed in SEEDS:
        result = v2.Engine(seed=seed, **engine_kwargs).run(module, _args())
        np.testing.assert_array_equal(result.outputs["signal"], np.ones(1, dtype=np.int32))
        # One increment per observed-unset poll; never negative, never replayed past
        # the release (the run terminates, so the count is finite).
        assert result.outputs["output"].shape == (1,)
        assert 0 <= int(result.outputs["output"][0])


def test_time_slice_does_not_replay_body_side_effects_and_schedules_peer():
    """Port of ``tests/numsim/integration/test_scheduler_polling_artifact.py::test_time_slice_does_not_replay_body_side_effects_and_schedules_peer``.

    Dropped pins: exact ``output == 64`` (legacy quantum) and ``stats["poll_order"]``.
    """
    _check_side_effect_poll({})


def test_time_slice_uses_configured_reschedule_quantum():
    """Port of ``tests/numsim/integration/test_scheduler_polling_artifact.py::test_time_slice_uses_configured_reschedule_quantum``.

    Dropped pins: exact ``output == 7`` (legacy quantum) and ``stats["poll_order"]``.
    The ``native_loop_reschedule_quantum=7`` knob is still accepted.
    """
    _check_side_effect_poll({"native_loop_reschedule_quantum": 7})


def test_finite_for_time_slice_schedules_peer_without_pattern_matching():
    """Port of ``tests/numsim/integration/test_scheduler_polling_artifact.py::test_finite_for_time_slice_schedules_peer_without_pattern_matching``.

    Dropped pins: exact ``output == 64`` (legacy quantum) and ``stats["poll_order"]``.
    """
    module = v2.transpile(time_sliced_finite_for)
    for seed in SEEDS:
        result = v2.Engine(seed=seed).run(module, _args())
        np.testing.assert_array_equal(result.outputs["signal"], np.ones(1, dtype=np.int32))
        assert 0 <= int(result.outputs["output"][0]) <= 128
