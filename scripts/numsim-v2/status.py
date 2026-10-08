"""One-command NumSim v2 migration status.

Usage (repository root)::

    python scripts/numsim-v2/status.py --run [-n 32]      # run the v2 conformance suite, then report
    python scripts/numsim-v2/status.py --junit results.xml   # report from an existing junit file

Prints, as Markdown:

* the conformance matrix per mode (``numsim``, ``racecheck``, ``synccheck``):
  ``match`` (v2 equals the legacy snapshot), ``delta-match`` (v2 equals a
  ``<mode>.delta.json`` naming a behaviour-delta row), ``no-oracle`` (the
  snapshot records a legacy exception; skipped under v2) and ``fail``;
* the public-API set from ``scripts/numsim-v2/coverage/v2_public_status.tsv``
  (functions and parametrized items passing under ``NUMSIM_IMPL=v2``).

``--run`` executes ``NUMSIM_IMPL=v2 pytest tests/conformance`` from
``tirx_harness/`` with the current interpreter (source scripts/dev-env.sh
first). The per-case triage in docs/development/v2-conformance-status.md is
produced separately; this script is the headline numbers.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAPSHOTS = REPO / "tirx_harness/tests/conformance/snapshots"
PUBLIC_TSV = REPO / "scripts/numsim-v2/coverage/v2_public_status.tsv"
MODES = ("numsim", "racecheck", "synccheck")
NODE = re.compile(r"test_conformance_snapshot\[(?P<case>.+)-(?P<mode>numsim|racecheck|synccheck)\]")


def run_suite(workers: int) -> Path:
    junit = Path(tempfile.mkstemp(prefix="numsim-v2-conformance-", suffix=".xml")[1])
    env = {**os.environ, "NUMSIM_IMPL": "v2"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-n", str(workers), "--dist=worksteal",
           "-p", "no:cacheprovider", "tests/conformance", f"--junitxml={junit}"]
    subprocess.run(cmd, cwd=REPO / "tirx_harness", env=env, check=False)
    return junit


def outcomes(junit: Path) -> dict[tuple[str, str], str]:
    out = {}
    for tc in ET.parse(junit).iter("testcase"):
        match = NODE.search(tc.get("name", ""))
        if not match:
            continue
        if tc.find("skipped") is not None:
            state = "skipped"
        elif tc.find("failure") is not None or tc.find("error") is not None:
            state = "failed"
        else:
            state = "passed"
        out[(match["case"], match["mode"])] = state
    return out


def classify(case: str, mode: str, state: str | None) -> str:
    base = SNAPSHOTS / case / f"{mode}.json"
    delta = SNAPSHOTS / case / f"{mode}.delta.json"
    legacy_error = base.exists() and "error" in json.loads(base.read_text())
    if state == "passed":
        return "delta-match" if delta.exists() else "match"
    if state == "skipped" and legacy_error and not delta.exists():
        return "no-oracle"
    return "fail" if state == "failed" else (state or "not-run")


def public_counts() -> tuple[int, int, int, int]:
    funcs = passing = items = items_pass = 0
    with PUBLIC_TSV.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["v2_status"] == "skip":
                continue
            funcs += 1
            passing += row["v2_status"] == "pass"
            for part in (row.get("items") or "").split():
                name, _, count = part.partition("=")
                if name == "skip":
                    continue
                items += int(count or 0)
                items_pass += int(count or 0) if name == "pass" else 0
    return passing, funcs, items_pass, items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="store_true", help="run the v2 conformance suite first")
    source.add_argument("--junit", type=Path, help="junit xml of a NUMSIM_IMPL=v2 conformance run")
    parser.add_argument("-n", type=int, default=32, help="pytest workers for --run")
    args = parser.parse_args()

    junit = run_suite(args.n) if args.run else args.junit
    states = outcomes(junit)
    cases = sorted(p.name for p in SNAPSHOTS.iterdir() if p.is_dir())
    table = {mode: collections.Counter() for mode in MODES}
    failing = []
    for case in cases:
        for mode in MODES:
            label = classify(case, mode, states.get((case, mode)))
            table[mode][label] += 1
            if label not in ("match", "delta-match", "no-oracle"):
                failing.append(f"{case}/{mode}: {label}")

    columns = ("match", "delta-match", "no-oracle", "fail")
    print(f"## Conformance (NUMSIM_IMPL=v2, {len(cases)} cases)\n")
    print("| mode | " + " | ".join(columns) + " |")
    print("| --- | " + " | ".join("---" for _ in columns) + " |")
    for mode in MODES:
        extra = sum(n for k, n in table[mode].items() if k not in columns)
        cells = [str(table[mode][c] + (extra if c == "fail" else 0)) for c in columns]
        print(f"| {mode} | " + " | ".join(cells) + " |")
    if failing:
        print("\nNot matching: " + ", ".join(failing))
    passing, funcs, items_pass, items = public_counts()
    print("\n## Public-API legacy tests under v2\n")
    print(f"{items_pass} of {items} items pass ({passing} of {funcs} functions fully), "
          f"from `{PUBLIC_TSV.relative_to(REPO)}`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
