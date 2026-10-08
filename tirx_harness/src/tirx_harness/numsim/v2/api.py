"""Public surface of NumSim v2 (plan 2.7): the legacy names, new engine.

``transpile``, ``Engine`` (``run``/``run_racecheck_phase``/
``run_synccheck_phase``), ``compare``, ``racecheck``, ``synccheck``,
``CoverageBounds``, ``ResourceLimits`` and the report types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compile import CompiledModule, ModuleContractError, from_document, transpile
from .report import (
    SCHEMA_VERSION,
    AnalysisResult,
    Finding,
    NumSimReport,
    NumSimResult,
    RaceReport,
    SyncCheckReport,
    compare,
    payload_json_schema,
)
from .run import (
    Engine,
    ExecutionError,
    ExecutionSubset,
    ExecutionSubsetSelection,
    InputError,
    MissingBindingsError,
    canonicalize_inputs,
)


def _nonnegative(owner: Any) -> None:
    for name, value in vars(owner).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{type(owner).__name__}.{name} must be a non-negative integer")


@dataclass(frozen=True)
class CoverageBounds:
    max_warp_preemptions: int
    max_completion_schedule_deviations: int

    def __post_init__(self) -> None:
        _nonnegative(self)


@dataclass(frozen=True)
class ResourceLimits:
    max_schedules: int
    max_backtrack_nodes: int
    max_events_per_run: int
    max_total_events: int
    max_loop_steps: int
    max_wall_time_ms: int
    max_diagnostic_bytes: int

    def __post_init__(self) -> None:
        _nonnegative(self)


def run_case(case: Any, *, engine: Engine | None = None) -> NumSimReport:
    """Transpile, run and compare one ``numsim.NumSimCase`` (legacy surface).

    As legacy ``run_case``: the host arrays bound to the kernel are frozen
    before ``case.reference()`` runs and restored afterwards, so a reference
    that mutates its inputs neither changes what the kernel sees nor leaks
    into the caller's arrays. The reference must be non-empty and may name
    only selected kernel outputs; both are checked before execution.
    """

    import copy

    import numpy as np

    from tirx_harness.numsim.errors import NumSimExecutionError

    from .run import _select_outputs, canonicalize_inputs

    engine = engine or Engine()
    module = transpile(case.kernel)
    selected = {
        external
        for _, external, _ in _select_outputs(
            canonicalize_inputs(module, case.args), case.outputs, module
        )
    }
    frozen: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[int] = set()
    for value in case.args.values():
        for array in (value, getattr(value, "_tensor_map_base", None)):
            if isinstance(array, np.ndarray) and id(array) not in seen:
                seen.add(id(array))
                frozen.append((array, array.copy()))
    try:
        expected = copy.deepcopy(case.reference())
    finally:
        for array, snapshot in frozen:
            if array.flags.writeable:
                np.copyto(array, snapshot)
    if not expected:
        raise NumSimExecutionError("NumSim expected outputs must not be empty")
    if not set(expected) <= selected:
        raise NumSimExecutionError(
            "NumSim reference outputs must name selected kernel outputs: "
            f"selected={sorted(selected)}, reference={sorted(expected)}"
        )
    result = engine.run(
        module, case.args, subset=case.subset, assumptions=case.assumptions, outputs=case.outputs
    )
    return compare(result, expected, tolerances=case.comparisons)


def _incomplete_phase(
    kind: str, name: str, reason: str, message: str, **details: Any
) -> AnalysisResult:
    """A typed fail-closed result for an invocation that cannot run (legacy
    returned these instead of raising)."""

    from .report import SCHEMA_VERSION as _schema

    record = {
        "kind": "analysis_incomplete",
        "status": "incomplete",
        "reason": reason,
        "message": message,
        **details,
    }
    payload = {
        "schema_version": _schema,
        "checker": kind,
        "engine": "numsim-core",
        "phase": {"index": 0, "name": name, "kernel_index": 0},
        "analysis_scope": {"kind": "full_launch"},
        "verdict": "incomplete",
        "findings": [],
        "advisories": [],
        "incomplete": [record],
        "execution_error": None,
        "stats": {"available": False},
        "coverage": {
            "status": "not_started",
            "eligible_for_clean": False,
            "termination": {"kind": reason},
        },
    }
    return AnalysisResult(kind, payload)


def _phases(kind: str, kernel: Any, inputs: dict | None, **kwargs: Any) -> list[AnalysisResult]:
    from tirx_harness.numsim.errors import UnsupportedTIRxError

    try:
        name = str(kernel.attrs["global_symbol"])
    except Exception:
        name = "kernel"
    try:
        module = transpile(kernel)
    except UnsupportedTIRxError as error:
        return [
            _incomplete_phase(
                kind,
                name,
                "native_frontend_unsupported",
                str(error),
                unsupported=list(getattr(error, "unsupported", ()) or ()),
            )
        ]
    engine = Engine()
    run_phase = getattr(engine, f"run_{kind}_phase")
    try:
        return [
            run_phase(module, dict(inputs or {}), phase_index=index, advance_prefix=True, **kwargs)
            for index in range(len(module.spec.kernels))
        ]
    except MissingBindingsError as error:
        return [
            _incomplete_phase(
                kind,
                module.spec.kernels[0].name,
                "missing_input_bindings",
                "native analysis requires complete concrete bindings before execution",
                bindings=error.missing,
            )
        ]


def racecheck(kernel: Any, inputs: dict | None = None) -> RaceReport:
    """Run Racecheck over every launch of one concrete invocation."""

    return RaceReport(_phases("racecheck", kernel, inputs))


def synccheck(
    kernel: Any,
    inputs: dict | None = None,
    *,
    coverage_bounds: CoverageBounds | None = None,
    resource_limits: ResourceLimits | None = None,
) -> SyncCheckReport:
    """Run Synccheck over every launch of one concrete invocation."""

    return SyncCheckReport(
        _phases(
            "synccheck",
            kernel,
            inputs,
            coverage_bounds=coverage_bounds,
            resource_limits=resource_limits,
        )
    )


__all__ = [
    "SCHEMA_VERSION",
    "AnalysisResult",
    "CompiledModule",
    "CoverageBounds",
    "Engine",
    "ExecutionError",
    "ExecutionSubset",
    "ExecutionSubsetSelection",
    "Finding",
    "InputError",
    "MissingBindingsError",
    "ModuleContractError",
    "NumSimReport",
    "NumSimResult",
    "RaceReport",
    "ResourceLimits",
    "SyncCheckReport",
    "canonicalize_inputs",
    "compare",
    "from_document",
    "payload_json_schema",
    "racecheck",
    "run_case",
    "synccheck",
    "transpile",
]
