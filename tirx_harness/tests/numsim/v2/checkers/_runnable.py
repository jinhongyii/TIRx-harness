"""Shared gate and assertion helpers for the v2 kernel-level checker tests.

These tests replace the 43 legacy checker tests that ``test-migration.md``
section B lists as ``gap_unportable``: their contract cannot be written as
contract events, so they must run a real TIRx kernel through lowering and the
interpreter. They only use the public v2 surface (``tirx_harness.numsim.v2``).

Every test carries :data:`requires_v2_engine`. It skips until the v2 engine
runs end to end, which :func:`v2_engine_runnable` probes once per process by
transpiling a vector add and running NumSim, Racecheck and Synccheck on it.
Set ``NUMSIM_V2_FORCE_CHECKERS=1`` to run the tests regardless (dev check:
shows how far each one gets today).
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterable

import numpy as np
import pytest

FORCE_ENV = "NUMSIM_V2_FORCE_CHECKERS"

_VECTOR_ADD = '''
@T.prim_func
def vadd(a: T.Buffer((256,), "float32"), b: T.Buffer((256,), "float32"),
         c: T.Buffer((256,), "float32")):
    T.device_entry()
    bx = T.cta_id([2])
    tx = T.thread_id([128])
    i = bx * 128 + tx
    if i < 256:
        c[i] = a[i] + b[i]
'''


@functools.lru_cache(maxsize=1)
def v2_engine_runnable() -> bool:
    """True only when the v2 engine runs a trivial kernel end to end."""

    if os.environ.get(FORCE_ENV, "") in {"1", "true", "yes"}:
        return True
    try:
        import tvm
        from tvm.script import tirx as T

        from tirx_harness.numsim import v2

        kernel = tvm.script.from_source(_VECTOR_ADD, {"T": T})
        a = np.arange(256, dtype=np.float32)
        b = np.full(256, 0.5, dtype=np.float32)

        def inputs():
            return {"a": a.copy(), "b": b.copy(), "c": np.zeros(256, dtype=np.float32)}

        module = v2.transpile(kernel)
        result = v2.Engine().run(module, inputs(), outputs=("c",))
        if result.status.get("kind") != "completed":
            return False
        if not np.array_equal(result.outputs["c"], a + b):
            return False
        if v2.racecheck(kernel, inputs()).verdict != "clean":
            return False
        if v2.synccheck(kernel, inputs()).verdict != "clean":
            return False
        return True
    except BaseException:  # noqa: BLE001 - any failure (incl. NotImplementedError, ImportError) means "not yet"
        return False


requires_v2_engine = pytest.mark.skipif(
    not v2_engine_runnable(),
    reason=f"v2 engine not runnable (set {FORCE_ENV}=1 to force)",
)


# -- report helpers (public report objects only) ------------------------------

_DETAIL_KIND_KEYS = ("kind", "legacy_kind", "source_kind", "reason")


def kinds_of(report) -> set[str]:
    """Every kind name a report's findings carry: the public ``kind`` plus the
    legacy/source kind the payload keeps in the finding details."""

    out: set[str] = set()
    for finding in report.findings:
        out.add(finding.kind)
        for key in _DETAIL_KIND_KEYS:
            value = finding.details.get(key)
            if isinstance(value, str):
                out.add(value)
    return out


def error_kinds_of(report) -> set[str]:
    out: set[str] = set()
    for finding in report.findings:
        if finding.status != "error":
            continue
        out.add(finding.kind)
        for key in _DETAIL_KIND_KEYS:
            value = finding.details.get(key)
            if isinstance(value, str):
                out.add(value)
    return out


def assert_clean(report) -> None:
    assert report.verdict == "clean", report.format()
    assert report.findings == [], report.format()


def assert_error_kind(report, kinds: Iterable[str]) -> None:
    """Verdict ``error`` and at least one error finding of one of ``kinds``."""

    wanted = set(kinds)
    assert report.verdict == "error", report.format()
    found = error_kinds_of(report)
    assert found & wanted, f"expected an error finding of kind {sorted(wanted)}, got {sorted(found)}\n{report.format()}"


def assert_no_incomplete(report) -> None:
    assert not [f for f in report.findings if f.status == "incomplete"], report.format()


def race_access_pairs(report) -> set[str]:
    return {
        str(f.details.get("access_pair"))
        for f in report.findings
        if f.status == "error" and f.details.get("access_pair") is not None
    }


# Kind names. ``OOB``: racecheck delta P5 (``execution_error{oob}`` ->
# ``OutOfBounds`` finding, public kind ``out_of_bounds``); the runtime
# diagnostic uses the same name in synccheck (no sync delta row).
OOB = frozenset({"out_of_bounds"})
RACE = frozenset({"data_race"})
DEADLOCK = frozenset({"deadlock"})
# Legacy ``warp_collective_divergence`` (an ``ExecError``). v2 reports it as the
# runtime diagnostic ``divergence`` (numsim-py ``exec_error_kind``); there is no
# delta row for the rename, so both spellings are accepted.
WARP_COLLECTIVE_DIVERGENCE = frozenset({"warp_collective_divergence", "divergence"})


def resource_limits(**overrides):
    from tirx_harness.numsim import v2

    values = dict(
        max_schedules=100,
        max_backtrack_nodes=10_000,
        max_events_per_run=10_000,
        max_total_events=100_000,
        max_loop_steps=100_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=1_000_000,
    )
    values.update(overrides)
    return v2.ResourceLimits(**values)


def coverage_bounds():
    from tirx_harness.numsim import v2

    return v2.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)


def no_spec(item: int, what: str):
    """``xfail(strict=False)`` for a legacy semantic that no new spec mentions
    (test-migration.md, "Semantics in legacy tests that no new spec mentions").
    The test still asserts the legacy expectation; it flips to a plain test
    once the ruling lands as a spec sentence or a delta row."""

    return pytest.mark.xfail(
        strict=False,
        reason=f"no spec: {what} (test-migration.md no-spec item {item}); ruling needed",
    )


def v2_gap(what: str):
    """``xfail(strict=False)`` for a faithful port that the current v2 engine
    or lowering gets wrong (observed when this suite was written). The test
    is correct; remove the marker once the gap is fixed (an XPASS says so)."""

    return pytest.mark.xfail(strict=False, reason=f"v2 gap: {what}")
