"""CI gate: a conformance snapshot may change only with a cited behaviour-delta row.

Usage (repository root)::

    python scripts/numsim-v2/check_snapshot_deltas.py --base origin/main [--head HEAD]

Rule (coordinator decision, 2026-10-08): every commit in ``base..head`` that
changes a file under ``tirx_harness/tests/conformance/snapshots/`` (a snapshot
or a ``<mode>.delta.json``) must cite at least one delta row id in its commit
message. A row id is ``[A-Z]{1,2}<digits>`` (e.g. ``B7``, ``T19``, ``S1``)
of one of the three behaviour-delta tables
(``docs/development/{numsim,racecheck,sync}-behaviour-deltas.md``).

Ids repeat across the tables, so citations are file-qualified: ``numsim T4``,
``racecheck B7``, ``sync M14`` (or the table's file name). A bare id is
accepted only when exactly one table has it; a commit that cites a bare id
present in several tables fails until it is qualified (``delta_rows.py``).

A whole-corpus regeneration that changes no verdict or finding (a snapshot
schema bump, a normalization change) instead carries the trailer
``Snapshot-Regen: schema <reason>``.

Changed cases the message does not name are reported as warnings (not
failures). Exit status 1 lists the offending commits.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAPSHOTS = "tirx_harness/tests/conformance/snapshots/"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_rows  # noqa: E402


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    ids = delta_rows.row_ids()
    failures = []
    for commit in git("rev-list", "--reverse", f"{args.base}..{args.head}").split():
        changed = [p for p in git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).split() if p.startswith(SNAPSHOTS)]
        if not changed:
            continue
        message = git("log", "-1", "--format=%B", commit)
        if re.search(r"^Snapshot-Regen: schema \S", message, re.M):
            continue
        citations = delta_rows.parse(message, ids)
        cited = sorted(citations.qualified)
        cases = sorted({Path(p).relative_to(SNAPSHOTS).parts[0] for p in changed})
        unnamed = [c for c in cases if c not in message]
        subject = git("log", "-1", "--format=%s", commit).strip()
        if citations.ambiguous or citations.unknown:
            problems = [f"ambiguous bare id(s) {', '.join(sorted(citations.ambiguous))} (qualify: numsim/racecheck/sync <id>)"] if citations.ambiguous else []
            problems += [f"unknown row(s) {', '.join(sorted(citations.unknown))}"] if citations.unknown else []
            failures.append(f"{commit[:12]} {subject}: " + "; ".join(problems))
        elif not cited:
            failures.append(f"{commit[:12]} {subject}: changes {len(changed)} snapshot file(s) "
                            f"({', '.join(cases[:6])}{' ...' if len(cases) > 6 else ''}) but cites no delta row id")
        elif unnamed:
            print(f"warning: {commit[:12]} {subject}: cites {', '.join(cited)} but does not name "
                  + ", ".join(unnamed[:8]) + (" ..." if len(unnamed) > 8 else ""))
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
