"""Where every category-A legacy test stands for redesign step 5.

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    # 1. public-API A tests under v2 -> coverage/v2_public_status.tsv
    NUMSIM_IMPL=v2 $PY -m pytest ... --junitxml=$TMP/public.xml <public A ids>
    $PY ../scripts/numsim-v2/v2_public_status.py $TMP/public.xml
    # 2. internal-surface A tests under v2 (same shim; their module imports still
    #    resolve while the legacy code exists)
    NUMSIM_IMPL=v2 $PY -m pytest ... --junitxml=$TMP/internal.xml <internal A ids>
    # 3. combine
    $PY ../scripts/numsim-v2/step5_status.py --internal $TMP/internal.xml [--v2-results JUNIT]

``--ids public|internal`` prints the node ids for steps 1 and 2.

Writes ``coverage/step5_a_status.tsv`` (``legacy_test``, ``surface``,
``v2_status``, ``step5``, ``legacy_internals_used``). ``step5`` is one of

* ``flip`` / ``flip-after-import-fix``: passes under v2 unchanged (internal
  files only need their module-level legacy import removed);
* ``retired-wave4``: a passing v2 copy exists (``coverage/v2_ports_*.tsv``);
* ``needs-port``: fails only on implementation pins (W9);
* ``uses-legacy-internals``: the test itself calls a legacy-only function
  (``analyze``, ``prepare_bindings``, ...): port or delete;
* ``blocked-v2``: v2 behaviour differs and nothing explains it yet (owner work);
* ``gpu-only``: skipped without a GPU (decided with the microtests);
* ``delete-C``: classified C after the run.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retire_tests  # noqa: E402
import v2_public_status  # noqa: E402

COVERAGE = HERE / "coverage"
BASE = retire_tests.TESTS_BASE
PUBLIC_MODULES = {"tirx_harness.numsim", "tirx_harness.numsim.api", "tirx_harness.numsim.errors"}


def legacy_names_used(test_id: str, cache: dict) -> set[str]:
    path, _, func = test_id.partition("::")
    func = func.split("::")[-1]
    if path not in cache:
        tree = ast.parse((BASE / path).read_text())
        internal = set()
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("tirx_harness.numsim")
                and node.module not in PUBLIC_MODULES
                and not node.module.startswith("tirx_harness.numsim.v2")
            ):
                internal |= {a.asname or a.name for a in node.names}
        defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        cache[path] = (internal, defs)
    internal, defs = cache[path]
    seen, todo, used = set(), [func], set()
    while todo:
        name = todo.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        for node in ast.walk(defs[name]):
            if isinstance(node, ast.Name):
                if node.id in internal:
                    used.add(node.id)
                elif node.id in defs:
                    todo.append(node.id)
    return used


def junit_status(path: Path) -> dict[str, str]:
    per: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for case in ET.parse(path).iter("testcase"):
        fid = "/".join(case.get("classname", "").split(".")) + ".py::" + case.get("name", "").split("[")[0]
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if case.find("skipped") is not None:
            per[fid]["skip"] += 1
        elif failure is None:
            per[fid]["pass"] += 1
        else:
            per[fid][v2_public_status.classify(failure.get("message") or "", failure.text or "")] += 1
    out = {}
    for fid, counts in per.items():
        fails = sorted(k for k in counts if k not in ("pass", "skip"))
        out[fid] = "+".join(fails) if fails else ("pass" if counts.get("pass") else "skip")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--internal", type=Path, help="junit XML of the internal-surface A run under v2")
    parser.add_argument("--v2-results", type=Path, help="junit XML of the v2 copies (else they are run)")
    parser.add_argument("--ids", choices=["public", "internal"])
    args = parser.parse_args()
    rows = list(csv.DictReader((COVERAGE / "test_classification.csv").open()))
    category = {r["test_id"]: r["category"] for r in rows}
    a_rows = [r for r in rows if r["category"] == "A" and r["target"] != "conformance"]
    if args.ids:
        want = "public" if args.ids == "public" else None
        for r in a_rows:
            if (r["surface"] == "public") == (want == "public"):
                print(r["test_id"])
        return 0
    if args.internal is None:
        parser.error("--internal is required")

    retired, _ = retire_tests.selection("4", args.v2_results)
    public = {r["legacy_test"]: r["v2_status"] for r in csv.DictReader((COVERAGE / "v2_public_status.tsv").open(), delimiter="\t")}
    internal = junit_status(args.internal)
    cache: dict = {}
    out = []
    for r in a_rows:
        test = r["test_id"]
        surface = "public" if r["surface"] == "public" else "internal"
        status = public.get(test, "") if surface == "public" else internal.get(test, "")
        used = set() if surface == "public" else legacy_names_used(test, cache)
        fails = [k for k in status.split("+") if k and k not in ("pass", "skip")]
        if category.get(test) == "C":
            step = "delete-C"
        elif test in retired:
            step = "retired-wave4"
        elif used:
            step = "uses-legacy-internals"
        elif status == "pass":
            step = "flip" if surface == "public" else "flip-after-import-fix"
        elif status == "skip":
            step = "gpu-only"
        elif fails and all(k.startswith("pin") or k == "other-assertion:port" for k in fails):
            step = "needs-port"
        else:
            step = "blocked-v2"
        out.append((test, surface, status, step, ",".join(sorted(used))))
    with (COVERAGE / "step5_a_status.tsv").open("w") as handle:
        handle.write("legacy_test\tsurface\tv2_status\tstep5\tlegacy_internals_used\n")
        for row in sorted(out):
            handle.write("\t".join(row) + "\n")
    counts = collections.Counter((row[1], row[3]) for row in out)
    for (surface, step), n in sorted(counts.items()):
        print(f"{surface:8s} {step:24s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
