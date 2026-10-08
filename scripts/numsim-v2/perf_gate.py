"""NumSim v2 performance gate: corpus workloads vs relative baselines.

Usage (repository root, after ``source scripts/dev-env.sh``)::

    $PY scripts/numsim-v2/perf_gate.py --run -n 1 [-k EXPR] [--record]
    $PY scripts/numsim-v2/perf_gate.py --junit results.xml [--record]

``--run`` executes ``tests/perf/test_corpus_perf.py`` (``-m performance``)
from ``tirx_harness/`` with ``NUMSIM_PERF_ENFORCE=0``, so every test records
its timed seconds (junit property ``perf_metrics``, metric -> seconds)
without asserting; this script applies the policy:

* a metric slower than ``max(baseline * (1 + tolerance), baseline +
  min_slack_seconds)`` (tolerance 0.5, i.e. 1.5x, and 0.1 s of slack for
  sub-second metrics, both from the baseline file) **fails** the gate when the host was near idle
  for the whole run (1-minute load average at most ``--idle-fraction`` of the
  CPU count, sampled before, during and after) and only **warns** otherwise;
* a correctness failure in a performance test always fails the gate;
* a metric with no baseline warns.

The load averages are printed. ``--record`` merges the measured seconds into the
baseline file ``tirx_harness/tests/perf/baselines/v2/<host-class>.json``
(refused under load unless ``--force``); review it and justify any increase
(tests/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "tirx_harness"
sys.path.insert(0, str(HARNESS))
from tests.perf import perf_baseline as pb  # noqa: E402

SUITE = "v2"
TEST_MODULE = "tests/perf/test_corpus_perf.py"
DEFAULT_TOLERANCE = 0.5
# Sub-second metrics: a regression must also exceed the baseline by this much.
DEFAULT_MIN_SLACK = 0.1


def load1() -> float:
    return os.getloadavg()[0]


class LoadSampler:
    """Samples the 1-minute load average every ``interval`` seconds."""

    def __init__(self, interval: float = 15.0):
        self.samples = [load1()]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)

    def _loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self.samples.append(load1())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.samples.append(load1())


def run_suite(workers: int, keyword: str | None) -> Path:
    junit = Path(tempfile.mkstemp(prefix="numsim-v2-perf-", suffix=".xml")[1])
    env = {**os.environ, "NUMSIM_PERF_ENFORCE": "0"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-n", str(workers), "-p", "no:cacheprovider",
           "-m", "performance", TEST_MODULE, f"--junitxml={junit}"]
    if keyword:
        cmd += ["-k", keyword]
    subprocess.run(cmd, cwd=HARNESS, env=env, check=False)
    return junit


def measurements(junit: Path) -> tuple[dict[str, list[float]], list[str]]:
    seconds: dict[str, list[float]] = {}
    failures = []
    for tc in ET.parse(junit).iter("testcase"):
        props = {p.get("name"): p.get("value") for p in tc.iter("property")}
        for metric, value in json.loads(props.get("perf_metrics") or "{}").items():
            seconds.setdefault(metric, []).append(float(value))
        failed = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if failed is not None:
            failures.append(f"{tc.get('name')}: {(failed.get('message') or '').splitlines()[0][:200]}")
    return seconds, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="store_true", help="run the performance tests first")
    source.add_argument("--junit", type=Path, help="junit xml of a NUMSIM_PERF_ENFORCE=0 run")
    parser.add_argument("-n", type=int, default=1, help="pytest workers for --run (engines use 1-32 threads)")
    parser.add_argument("-k", dest="keyword", help="pytest -k selection for --run")
    parser.add_argument("--idle-fraction", type=float, default=0.10,
                        help="near idle: 1-minute load average <= this fraction of the CPU count")
    parser.add_argument("--record", action="store_true", help="write the measurements as the baseline")
    parser.add_argument("--force", action="store_true", help="--record even under load")
    args = parser.parse_args()

    nproc = os.cpu_count() or 1
    if args.run:
        with LoadSampler() as sampler:
            junit = run_suite(args.n, args.keyword)
        loads = sampler.samples
    else:
        junit, loads = args.junit, [load1()]
    idle_limit = args.idle_fraction * nproc
    idle = max(loads) <= idle_limit
    host = pb.host_class()
    print(f"host class {host}, {nproc} CPUs; 1-min load average min/max {min(loads):.1f}/{max(loads):.1f} "
          f"over {len(loads)} samples (near-idle limit {idle_limit:.1f}): {'near idle' if idle else 'UNDER LOAD'}")

    seconds, failures = measurements(junit)
    try:
        data = pb.load_baselines(host, SUITE)
    except pb.BaselineMissing:
        data = {"baselines": {}}
    file_tolerance = float(data.get("tolerance", DEFAULT_TOLERANCE))
    file_slack = float(data.get("min_slack_seconds", DEFAULT_MIN_SLACK))
    regressions, warnings = [], []
    print("\n| metric | seconds | baseline | ratio | limit |\n| --- | --- | --- | --- | --- |")
    for metric in sorted(seconds):
        value = max(seconds[metric])
        entry = data["baselines"].get(metric)
        if entry is None:
            print(f"| {metric} | {value:.2f} | - | - | - |")
            warnings.append(f"{metric}: no baseline")
            continue
        expected = pb.Baseline(metric, float(entry["seconds"]), float(entry.get("tolerance", file_tolerance)),
                               float(entry.get("min_slack_seconds", file_slack)))
        base, tolerance = expected.seconds, expected.tolerance
        ratio = value / base if base else float("inf")
        print(f"| {metric} | {value:.3f} | {base:.3f} | {ratio:.2f}x | {expected.limit:.3f}s |")
        if value > expected.limit:
            (regressions if idle else warnings).append(
                f"{metric}: {value:.2f}s is {ratio:.2f}x the baseline {base:.2f}s (limit {1 + tolerance:.2f}x)"
                + ("" if idle else " [under load: warning only]"))
    for line in warnings:
        print(f"WARNING: {line}")
    for line in failures:
        print(f"FAILED: {line}")
    for line in regressions:
        print(f"REGRESSION: {line}")

    if args.record:
        if not idle and not args.force:
            print("not recording: the host was under load (pass --force to record anyway)")
            return 1
        path = pb.baseline_path(host, SUITE)
        path.parent.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, check=False).stdout.strip()
        record = {
            "host_class": host,
            "cpu_model": pb.cpu_model(),
            "nproc": nproc,
            "recorded": datetime.date.today().isoformat(),
            "commit": f"{commit} + working tree",
            "preflight": f"1-min load average min/max {min(loads):.1f}/{max(loads):.1f} over the run",
            "pytest": f"-n {args.n} -m performance {TEST_MODULE}"
            + (f" -k {args.keyword!r}" if args.keyword else ""),
            "tolerance": file_tolerance,
            "min_slack_seconds": file_slack,
            "baselines": dict(data.get("baselines") or {}),
        }
        # Merge: a partial run (-k) updates only the metrics it measured.
        for metric, values in sorted(seconds.items()):
            record["baselines"][metric] = {
                "seconds": round(max(values), 3),
                "samples": [round(v, 3) for v in values],
                "preflight": record["preflight"],
                "recorded": record["recorded"],
            }
        record["baselines"] = dict(sorted(record["baselines"].items()))
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"recorded {len(seconds)} metrics in {path}")

    return 1 if (failures or regressions) else 0


if __name__ == "__main__":
    sys.exit(main())
