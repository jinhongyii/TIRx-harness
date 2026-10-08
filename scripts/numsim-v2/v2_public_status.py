"""Summarize a ``NUMSIM_IMPL=v2`` run of the public-API A tests per function.

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    NUMSIM_IMPL=v2 $PY -m pytest -q -n 32 --dist=worksteal -m 'not numsim_gpu' \\
        --junitxml=$TMP/v2_public.xml $(public-API A node ids from test_classification.csv)
    $PY ../scripts/numsim-v2/v2_public_status.py $TMP/v2_public.xml

Writes ``coverage/v2_public_status.tsv`` (``legacy_test``, ``v2_status``,
``items``). ``v2_status`` is ``pass``, ``skip`` or ``+``-joined failure
classes: ``pin-internals``, ``pin-message``, ``pin-message(error->incomplete)``,
``v2-accepts-legacy-rejection``, ``racecheck-verdict``, ``synccheck-verdict``,
``lowering-rejects``, ``engine-stops``, ``other-assertion`` (refined to
``other-assertion:{port,delta,bug}`` by ``coverage/other_assertion_triage.tsv``).
"""

from __future__ import annotations

import collections
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

COVERAGE = Path(__file__).resolve().parent / "coverage"

PIN = re.compile(
    r"KeyError: '(task_count|completed_task_count|poll_order|worker_count|tmap)'"
    r"|has no attribute '(rust_source|load|semantic_requirements)'|Hint:|in set\(\)"
)


def classify(message: str, text: str) -> str:
    if PIN.search(message):
        return "pin-internals"
    if "Regex pattern did not match" in message:
        actual = re.search(r"Actual message: (.*)", text or "")
        return "pin-message(error->incomplete)" if actual and "incomplete" in actual.group(1) else "pin-message"
    if "DID NOT RAISE" in message:
        return "v2-accepts-legacy-rejection"
    if re.search(r"racecheck (ERROR|REVIEW|INCOMPLETE|CLEAN)", message):
        return "racecheck-verdict"
    if re.search(r"synccheck (ERROR|REVIEW|INCOMPLETE|CLEAN)", message):
        return "synccheck-verdict"
    if "UnsupportedTIRxError" in message:
        return "lowering-rejects"
    if "ExecutionError" in message:
        return "engine-stops"
    return "other-assertion"


def main() -> int:
    results = Path(sys.argv[1])
    per_function: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for case in ET.parse(results).iter("testcase"):
        function = "/".join(case.get("classname", "").split(".")) + ".py::" + case.get("name", "").split("[")[0]
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if case.find("skipped") is not None:
            per_function[function]["skip"] += 1
        elif failure is None:
            per_function[function]["pass"] += 1
        else:
            per_function[function][classify(failure.get("message") or "", failure.text or "")] += 1
    # Hand triage of ``other-assertion`` failures (port / delta / bug).
    triage = {}
    triage_file = COVERAGE / "other_assertion_triage.tsv"
    if triage_file.exists():
        triage = {r["legacy_test"]: r["class"] for r in csv.DictReader(triage_file.open(), delimiter="\t")}
    rows = list(csv.DictReader((COVERAGE / "test_classification.csv").open()))
    public = [r["test_id"] for r in rows if r["category"] == "A" and r["surface"] == "public" and r["target"] != "conformance"]
    totals: collections.Counter = collections.Counter()
    with (COVERAGE / "v2_public_status.tsv").open("w") as out:
        out.write("legacy_test\tv2_status\titems\n")
        for test in public:
            counts = per_function.get(test, collections.Counter())
            fails = sorted(
                (f"other-assertion:{triage[test]}" if k == "other-assertion" and test in triage else k)
                for k in counts
                if k not in ("pass", "skip")
            )
            status = "+".join(fails) if fails else ("pass" if counts.get("pass") else "skip")
            out.write(f"{test}\t{status}\t{' '.join(f'{k}={v}' for k, v in sorted(counts.items()))}\n")
            totals[status.split("+")[0]] += 1
    print(", ".join(f"{k}: {v}" for k, v in totals.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
