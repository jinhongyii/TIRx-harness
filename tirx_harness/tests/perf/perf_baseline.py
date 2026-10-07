"""Relative performance baselines, one JSON file per host class.

A baseline file ``baselines/<host-class>.json`` records the measured wall
time of named workloads on one class of host. Tests assert that a fresh
measurement stays within ``baseline * (1 + tolerance)`` instead of comparing
against an absolute number that only holds on one machine.

The host class defaults to ``<cpu-model-slug>-<nproc>c`` (for example
``amd-epyc-7763-64-core-processor-256c``) and can be overridden with
``NUMSIM_PERF_HOST_CLASS``. A host without a baseline file skips the check;
see ``baselines/README.md`` for how to record one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
HOST_CLASS_ENV = "NUMSIM_PERF_HOST_CLASS"
DEFAULT_TOLERANCE = 0.10


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or platform.machine() or "unknown-cpu"


def host_class() -> str:
    explicit = os.environ.get(HOST_CLASS_ENV, "").strip()
    if explicit:
        return explicit
    slug = re.sub(r"[^a-z0-9]+", "-", cpu_model().lower()).strip("-")
    return f"{slug}-{os.cpu_count()}c"


@dataclass(frozen=True)
class Baseline:
    name: str
    seconds: float
    tolerance: float

    @property
    def limit(self) -> float:
        return self.seconds * (1.0 + self.tolerance)


class BaselineMissing(LookupError):
    pass


@lru_cache(maxsize=None)
def load_baselines(host: str | None = None) -> dict[str, Any]:
    host = host or host_class()
    path = BASELINE_DIR / f"{host}.json"
    if not path.exists():
        raise BaselineMissing(f"no performance baseline file for host class {host!r} ({path})")
    data = json.loads(path.read_text())
    if data.get("host_class") != host:
        raise ValueError(f"{path} declares host_class {data.get('host_class')!r}, expected {host!r}")
    return data


def baseline(name: str, *, host: str | None = None) -> Baseline:
    data = load_baselines(host)
    try:
        entry = data["baselines"][name]
    except KeyError as error:
        raise BaselineMissing(
            f"host class {data['host_class']!r} has no baseline named {name!r}"
        ) from error
    tolerance = float(entry.get("tolerance", data.get("tolerance", DEFAULT_TOLERANCE)))
    return Baseline(name=name, seconds=float(entry["seconds"]), tolerance=tolerance)


def assert_within_baseline(name: str, elapsed: float, *, host: str | None = None) -> Baseline:
    """Assert ``elapsed <= baseline * (1 + tolerance)``; skip without a baseline.

    Call this from a ``performance``-marked test after measuring only the
    timed region (exclude preparation and artifact builds).
    """

    import pytest

    try:
        expected = baseline(name, host=host)
    except BaselineMissing as error:
        pytest.skip(str(error))
    assert elapsed <= expected.limit, (
        f"{name}: {elapsed:.3f}s exceeds the {host or host_class()} baseline "
        f"{expected.seconds:.3f}s x (1 + {expected.tolerance:.2f}) = {expected.limit:.3f}s"
    )
    return expected


__all__ = [
    "BASELINE_DIR",
    "DEFAULT_TOLERANCE",
    "Baseline",
    "BaselineMissing",
    "assert_within_baseline",
    "baseline",
    "host_class",
    "load_baselines",
]
