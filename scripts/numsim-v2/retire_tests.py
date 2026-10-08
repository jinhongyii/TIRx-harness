"""Apply one wave of the legacy-test retirement plan (docs/development/test-migration.md).

Usage (repository root)::

    python scripts/numsim-v2/retire_tests.py --wave 0 --dry-run      # print the plan
    python scripts/numsim-v2/retire_tests.py --wave 1 --markdown     # the doc's list
    python scripts/numsim-v2/retire_tests.py --wave 0                # apply (NOT yet: legacy is the oracle)

Waves (rows of ``coverage/test_classification.csv``):

* ``0``  category E: tile forms TVM's own dispatch rejects.
* ``1``  category B with status ``covered`` or ``ported``: a Rust scenario
  test built from contract events replaces them.
* ``5b`` category C: implementation pins, deleted together with the legacy
  code at redesign step 5.

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
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CSV = HERE / "coverage" / "test_classification.csv"
TESTS_BASE = REPO / "tirx_harness"


def selected(row: dict[str, str], wave: str) -> bool:
    if wave == "0":
        return row["category"] == "E"
    if wave == "1":
        return row["category"] == "B" and row["target"] in {"covered", "ported"}
    if wave == "5b":
        return row["category"] == "C"
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


def plan(wave: str) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    rows = list(csv.DictReader(CSV.open()))
    chosen: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        if selected(row, wave):
            file, _, func = row["test_id"].partition("::")
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
    return whole, partial, missing


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
    parser.add_argument("--wave", required=True, choices=["0", "1", "5b"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--markdown", action="store_true", help="print the plan as Markdown and exit")
    args = parser.parse_args()
    whole, partial, missing = plan(args.wave)
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
