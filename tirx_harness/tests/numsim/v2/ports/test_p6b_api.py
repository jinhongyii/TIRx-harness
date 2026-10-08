"""v2 ports of the legacy ``run_case`` binding/reference-ordering tests.

Legacy: tests/numsim/integration/test_api.py drove ``numsim.run_case`` with a
monkeypatched ``transpile`` and a ``_FrozenNoOpEngine`` built on the legacy
private ``Engine._prepare_execution`` / ``_execute_prepared`` split and
``bindings.prepare_bindings``. Those internals have no v2 counterpart, so the
v2 copies use a real one-line kernel (``output[0] = source[0]``) and observe
the same contracts through the public surface: ``v2.run_case`` with an
``engine=`` that is a ``v2.Engine`` subclass recording whether ``run`` was
called. Dropped: the monkeypatched ``transpile`` and the legacy
``_PreparedExecution`` / ``prepare_bindings`` / ``apply_allocation_bytes``
plumbing of the fake engine.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2
from tirx_harness.numsim.cases import NumSimCase
from tirx_harness.numsim.errors import NumSimExecutionError

pytestmark = requires_v2_engine


@T.prim_func
def copy_source(source: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = source[0]


class _RecordingEngine(v2.Engine):
    """A real v2 engine that records whether ``run`` executed."""

    def __init__(self) -> None:
        super().__init__()
        self.executed = False

    def run(self, *args, **kwargs):
        self.executed = True
        return super().run(*args, **kwargs)


@v2_gap(
    "v2.run_case does not check reference keys: it runs the kernel (engine.run called) "
    "and returns a compare report without raising; legacy raised "
    "NumSimExecutionError('... must name selected kernel outputs ...') before execution"
)
def test_run_case_requires_reference_keys_to_name_selected_outputs():
    """Port of tests/numsim/integration/test_api.py::test_run_case_requires_reference_keys_to_name_selected_outputs.

    Dropped: the monkeypatched ``transpile`` and the legacy fake engine
    (see the module docstring).
    """

    engine = _RecordingEngine()
    case = NumSimCase(
        kernel=copy_source,
        args={
            "source": np.array([3], dtype=np.int32),
            "output": np.zeros(1, dtype=np.int32),
        },
        outputs=("output",),
        reference=lambda: {"wrong_name": np.zeros(1, dtype=np.int32)},
    )

    with pytest.raises(NumSimExecutionError, match="must name selected"):
        v2.run_case(case, engine=engine)

    assert not engine.executed


@v2_gap(
    "v2.run_case raises a plain ValueError('NumSim expected outputs must not be empty') "
    "for an empty reference; legacy raised NumSimExecutionError (not executed in both)"
)
def test_run_case_rejects_empty_reference_before_execution():
    """Port of tests/numsim/integration/test_api.py::test_run_case_rejects_empty_reference_before_execution.

    Dropped: the monkeypatched ``transpile`` and the legacy fake engine
    (see the module docstring).
    """

    engine = _RecordingEngine()
    case = NumSimCase(
        kernel=copy_source,
        args={
            "source": np.array([3], dtype=np.int32),
            "output": np.zeros(1, dtype=np.int32),
        },
        outputs=("output",),
        reference=lambda: {},
    )

    with pytest.raises(NumSimExecutionError, match="must not be empty"):
        v2.run_case(case, engine=engine)

    assert not engine.executed


@v2_gap(
    "v2.run_case calls the reference before binding inputs and never restores host "
    "buffers: the kernel sees the reference's source=99 (mismatch actual=99, "
    "expected the frozen 3) and the caller's source/output stay mutated (99/7) instead of restored (3/0)"
)
def test_run_case_freezes_bindings_before_mutating_reference():
    """Port of tests/numsim/integration/test_api.py::test_run_case_freezes_bindings_before_mutating_reference.

    The legacy no-op engine left ``output`` at its frozen 0 and recorded the
    ``source`` it observed; here the kernel copies ``source`` into ``output``,
    so the mismatch's ``actual`` is the frozen ``source`` (3) the engine saw
    (legacy: ``actual == 0`` plus ``observed_source == [3]``). Expected 7 and
    the host-buffer restoration (``source == 3``; ``output`` back to its frozen
    0, or the kernel's 3 if the engine writes outputs back) are kept.
    """

    source = np.array([3], dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    def mutating_reference():
        source[0] = 99
        output[0] = 7
        return {"output": output}

    case = NumSimCase(
        kernel=copy_source,
        args={"source": source, "output": output},
        outputs=("output",),
        reference=mutating_reference,
    )

    report = v2.run_case(case, engine=v2.Engine())

    assert not report.ok
    assert report.mismatches[0].actual == 3
    assert report.mismatches[0].expected == 7
    np.testing.assert_array_equal(source, np.array([3], dtype=np.int32))
    # The reference's 7 must not survive: ``output`` is restored to its frozen
    # 0. A real engine may then write the kernel's 3 back into the host array
    # (the legacy engine does, v2 does not; the legacy no-op engine left 0),
    # which is outside this contract.
    assert output[0] in (0, 3), output
