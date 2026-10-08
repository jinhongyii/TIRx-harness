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
from .run import Engine, ExecutionError, InputError, canonicalize_inputs


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
    """Transpile, run and compare one ``numsim.NumSimCase`` (legacy surface)."""

    import copy

    engine = engine or Engine()
    module = transpile(case.kernel)
    expected = copy.deepcopy(case.reference())
    if not expected:
        raise ValueError("NumSim expected outputs must not be empty")
    result = engine.run(
        module, case.args, subset=case.subset, assumptions=case.assumptions, outputs=case.outputs
    )
    return compare(result, expected, tolerances=case.comparisons)


def _phases(kind: str, kernel: Any, inputs: dict | None, **kwargs: Any) -> list[AnalysisResult]:
    module = transpile(kernel)
    engine = Engine()
    run_phase = getattr(engine, f"run_{kind}_phase")
    return [
        run_phase(module, dict(inputs or {}), phase_index=index, advance_prefix=True, **kwargs)
        for index in range(len(module.spec.kernels))
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
        _phases("synccheck", kernel, inputs, coverage_bounds=coverage_bounds, resource_limits=resource_limits)
    )


__all__ = [
    "SCHEMA_VERSION",
    "AnalysisResult",
    "CompiledModule",
    "CoverageBounds",
    "Engine",
    "ExecutionError",
    "Finding",
    "InputError",
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
