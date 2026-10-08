"""Apply one wave of the legacy-test retirement plan (docs/development/test-migration.md).

Usage (repository root)::

    python scripts/numsim-v2/retire_tests.py --wave 0 --dry-run      # print the plan
    python scripts/numsim-v2/retire_tests.py --wave 4 --dry-run --verbose   # runs the v2 copies first
    python scripts/numsim-v2/retire_tests.py --wave 1 --markdown     # the doc's list
    python scripts/numsim-v2/retire_tests.py --wave 0                # apply (NOT yet: legacy is the oracle)

Waves (rows of ``coverage/test_classification.csv``):

* ``0``  category E: tile forms TVM's own dispatch rejects.
* ``1``  category B with status ``covered`` or ``ported``: a Rust scenario
  test built from contract events replaces them.
* ``2``  category B with status ``v2_kernel_test``: retired once every v2
  replacement (``tests/numsim/v2/checkers``) passes.
* ``3``  category A with target ``conformance``: the corpus / wiki verdict
  gates that tests/conformance snapshots replace (run at deletion time).
* ``4``  category A, public-API surface: functions with a v2 copy in
  ``coverage/v2_ports_*.tsv`` are retired once every copy passes; functions
  that pass unchanged under ``NUMSIM_IMPL=v2`` (``v2_public_status.tsv``)
  are listed as ``flip`` (no edit: they become v2 tests when the public names
  point at v2).
* ``5b`` category C: implementation pins, deleted together with the legacy
  code at redesign step 5.

Waves 2 and 4 are gated on the replacements passing. The script runs them
(``pytest --junitxml``) unless ``--v2-results JUNIT.xml`` is given. A
function counts as passing when every collected item passed, including a
non-strict XPASS, or a strict xfail whose reason says "documented limitation"
(a delta row records the gap); any other failure, error, xfail or skip keeps
the legacy test.

A file whose every test function is selected is removed with ``git rm``;
otherwise the selected functions (with their decorators) are cut out of the
file. Module helpers and imports that become unused are left for ``ruff`` to
report; the script lists the files that need that follow-up.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COVERAGE = HERE / "coverage"
CSV = COVERAGE / "test_classification.csv"
TESTS_BASE = REPO / "tirx_harness"


def replacements(wave: str, rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """legacy function id -> v2 replacement node ids (function level)."""
    out: dict[str, list[str]] = {}
    if wave == "2":
        for row in rows:
            if row["category"] == "B" and row["target"] == "v2_kernel_test":
                out[row["test_id"]] = [t for t in row["rust_tests"].split(";") if t.startswith("tests/numsim/v2/")]
    elif wave == "4":
        for path in sorted(COVERAGE.glob("v2_ports_*.tsv")):
            with path.open() as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    tests = [t for t in row["v2_tests"].split(";") if t.strip()]
                    if tests:
                        # Union across port maps: a legacy function may be covered
                        # by copies listed in several files (one per porter).
                        merged = out.setdefault(row["legacy_test"], [])
                        merged += [t.strip() for t in tests if t.strip() not in merged]
    return out


def gpu_marked(func_id: str) -> bool:
    """True when the replacement function carries ``@pytest.mark.numsim_gpu``.

    Coordinator ruling (after e192f28): GPU-oracle copies form the gpu-only
    bucket. They skip off a GPU host and run on one, so a skip does not hold
    the legacy cut."""
    path = TESTS_BASE / func_id.split("::", 1)[0]
    name = func_id.split("::")[-1]
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return any("numsim_gpu" in ast.unparse(d) for d in node.decorator_list)
    return False


def passing(node_ids: set[str], results: Path | None) -> set[str]:
    """Function-level node ids whose every item passed."""
    if not node_ids:
        return set()
    if results is None:
        handle, name = tempfile.mkstemp(suffix=".xml")
        os.close(handle)
        results = Path(name)
        cmd = [sys.executable, "-m", "pytest", "-q", "-n", "16", "-p", "no:cacheprovider", f"--junitxml={results}"]
        # Run whole files: one stale node id would abort a node-id run and
        # make every replacement look failed.
        files = sorted({n.split("::", 1)[0] for n in node_ids if (TESTS_BASE / n.split("::", 1)[0]).exists()})
        subprocess.run(cmd + files, cwd=TESTS_BASE, stdout=subprocess.DEVNULL, check=False)
    outcome: dict[str, bool] = {}
    cases = list(ET.parse(results).iter("testcase"))
    if not cases:
        raise SystemExit(f"no test results in {results}: the replacement run failed to collect; rerun")
    for case in cases:
        file = case.get("classname", "").replace(".", "/")
        name = case.get("name", "").split("[")[0]
        # classname is ``tests.x.y.test_mod`` or ``tests.x.y.test_mod.TestClass``
        parts = file.split("/")
        if parts[-1].startswith("Test"):
            func = f"{'/'.join(parts[:-1])}.py::{parts[-1]}::{name}"
        else:
            func = f"{file}.py::{name}"
        skipped = case.find("skipped")
        # A strict xfail that names a documented limitation (e.g. racecheck
        # delta T18) is the accepted contract, so it counts as passing.
        documented = skipped is not None and "documented limitation" in (skipped.get("message") or "")
        ok = all(case.find(tag) is None for tag in ("failure", "error")) and (skipped is None or documented)
        if not ok and skipped is not None and all(case.find(t) is None for t in ("failure", "error")):
            ok = gpu_marked(func)
        outcome[func] = outcome.get(func, True) and ok
    return {f for f in node_ids if outcome.get(f)}


def selection(wave: str, results: Path | None) -> tuple[set[str], dict[str, str]]:
    """Function ids to retire, plus informational rows (id -> reason)."""
    rows = list(csv.DictReader(CSV.open()))
    info: dict[str, str] = {}
    if wave == "0":
        return {r["test_id"] for r in rows if r["category"] == "E"}, info
    if wave == "3":
        # Corpus / wiki verdict gates: tests/conformance snapshots cover them.
        return {r["test_id"] for r in rows if r["category"] == "A" and r["target"] == "conformance"}, info
    if wave == "1":
        return {r["test_id"] for r in rows if r["category"] == "B" and r["target"] in {"covered", "ported"}}, info
    if wave == "5b":
        return {r["test_id"] for r in rows if r["category"] == "C"}, info
    if wave in {"2", "4"}:
        repl = replacements(wave, rows)
        ok = passing({t for tests in repl.values() for t in tests}, results)
        chosen = set()
        for legacy, tests in repl.items():
            if tests and all(t in ok for t in tests):
                chosen.add(legacy)
            else:
                info[legacy] = "hold: replacement not passing (" + ", ".join(t for t in tests if t not in ok)[:200] + ")"
        if wave == "4":
            for r in rows:
                if r["category"] == "A" and r["surface"] == "public" and r["test_id"] not in repl:
                    if r.get("v2_status") == "pass":
                        info[r["test_id"]] = "flip: passes unchanged under NUMSIM_IMPL=v2"
                    elif r.get("v2_status"):
                        info[r["test_id"]] = f"hold: {r['v2_status']} under NUMSIM_IMPL=v2, no v2 copy"
        return chosen, info
    raise SystemExit(f"wave {wave} has no mechanical selection (see test-migration.md)")


def test_functions(path: Path) -> dict[str, tuple[int, int]]:
    """``Class::name`` or ``name`` -> (first line incl. decorators, last line), 1-based."""
    tree = ast.parse(path.read_text())
    out: dict[str, tuple[int, int]] = {}

    def span(node: ast.AST) -> tuple[int, int]:
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return first, node.end_lineno

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            out[node.name] = span(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test_"):
                    out[f"{node.name}::{sub.name}"] = span(sub)
    return out


def plan(wave: str, results: Path | None = None):
    ids, info = selection(wave, results)
    chosen: dict[str, set[str]] = collections.defaultdict(set)
    for test_id in ids:
        file, _, func = test_id.partition("::")
        chosen[file].add(func)
    whole: list[str] = []
    partial: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for file in sorted(chosen):
        path = TESTS_BASE / file
        if not path.exists():
            missing[file] = sorted(chosen[file])
            continue
        funcs = test_functions(path)
        gone = sorted(set(chosen[file]) - funcs.keys())
        if gone:
            missing[file] = gone
        live = chosen[file] & funcs.keys()
        if live and live == set(funcs):
            whole.append(file)
        elif live:
            partial[file] = sorted(live, key=lambda f: funcs[f][0])
    return whole, partial, missing, info


def cut(path: Path, funcs: list[str]) -> None:
    spans = test_functions(path)
    lines = path.read_text().splitlines(keepends=True)
    for func in sorted(funcs, key=lambda f: spans[f][0], reverse=True):
        first, last = spans[func]
        # Swallow the blank lines that separated the function from its successor.
        while last < len(lines) and not lines[last].strip():
            last += 1
        del lines[first - 1 : last]
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wave", required=True, choices=["0", "1", "2", "3", "4", "5b"])
    parser.add_argument("--v2-results", type=Path, help="junit XML of the v2 replacements (waves 2 and 4)")
    parser.add_argument("--verbose", action="store_true", help="also print flip / hold rows")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--markdown", action="store_true", help="print the plan as Markdown and exit")
    args = parser.parse_args()
    whole, partial, missing, info = plan(args.wave, args.v2_results)
    n_funcs = sum(len(v) for v in partial.values())
    n_whole = sum(len(test_functions(TESTS_BASE / f)) for f in whole)

    if args.markdown:
        print(f"Wave {args.wave}: {len(whole)} whole files ({n_whole} tests), "
              f"{n_funcs} functions cut from {len(partial)} files.\n")
        print("```bash")
        print("cd tirx_harness")
        for file in whole:
            print(f"git rm {file}")
        print("```\n")
        for file, funcs in partial.items():
            print(f"- `{file}`: " + ", ".join(f"`{f}`" for f in funcs))
        return 0

    for file in whole:
        print(f"git rm tirx_harness/{file}")
    for file, funcs in partial.items():
        for func in funcs:
            print(f"cut    tirx_harness/{file}::{func}")
    kinds = collections.Counter(reason.split(":")[0] for reason in info.values())
    if args.verbose:
        for test_id, reason in sorted(info.items()):
            print(f"# {reason.split(':')[0]:5s} tirx_harness/{test_id}  ({reason.split(':', 1)[1].strip()})")
    if kinds:
        print("# " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())), file=sys.stderr)
    for file, funcs in missing.items():
        print(f"warning: not found (stale classification?): {file}: {', '.join(funcs)}", file=sys.stderr)
    print(
        f"# wave {args.wave}: {len(whole)} files ({n_whole} tests) removed, "
        f"{n_funcs} functions cut from {len(partial)} files",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    for file in whole:
        subprocess.run(["git", "rm", "-q", str(TESTS_BASE / file)], check=True, cwd=REPO)
    for file, funcs in partial.items():
        cut(TESTS_BASE / file, funcs)
    if partial:
        print("follow-up: run `ruff check --select F401,F811,F841` on the edited files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
