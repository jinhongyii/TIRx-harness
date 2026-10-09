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
``Snapshot-Regen: schema <reason>`` or ``Snapshot-Regen: harness <reason>``
(a test-harness change, e.g. host-independent input generation, that moves
output bits but no verdict, finding or coverage).

Changed cases the message does not name are reported as warnings (not
failures). Exit status 1 lists the offending commits.

The rule is enforced from ``ENFORCED_FROM`` (506c6e6, the commit that made
citations file-qualified) on: commits that are ancestors of it predate the
rule (the initial snapshot generation, schema bumps and bare-id delta files
of the migration) and are skipped, so a PR or push range that still contains
that history (e.g. the refactor branch against an older main) is judged only
on its commits since the rule existed. ``--enforced-from`` overrides it.
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


# The commit that introduced file-qualified citations (2026-10-08).
ENFORCED_FROM = "506c6e6ed43a1e6ae536e506f9f481e2d699e8e6"


def judge(message: str, changed: list[str], ids: dict[str, set[str]]) -> tuple[str | None, str | None]:
    """``(failure, warning)`` for one commit that changes ``changed`` snapshot
    paths (relative to the repository root) with commit message ``message``."""

    if re.search(r"^Snapshot-Regen: (schema|harness) \S", message, re.M):
        return None, None
    citations = delta_rows.parse(message, ids)
    cited = sorted(citations.qualified)
    cases = sorted({Path(p).relative_to(SNAPSHOTS).parts[0] for p in changed})
    if citations.ambiguous or citations.unknown:
        problems = []
        if citations.ambiguous:
            problems.append(f"ambiguous bare id(s) {', '.join(sorted(citations.ambiguous))} "
                            "(qualify: numsim/racecheck/sync <id>)")
        if citations.unknown:
            problems.append(f"unknown row(s) {', '.join(sorted(citations.unknown))}")
        return "; ".join(problems), None
    if not cited:
        return (f"changes {len(changed)} snapshot file(s) "
                f"({', '.join(cases[:6])}{' ...' if len(cases) > 6 else ''}) but cites no delta row id"), None
    unnamed = [c for c in cases if c not in message]
    if unnamed:
        return None, (f"cites {', '.join(cited)} but does not name "
                      + ", ".join(unnamed[:8]) + (" ..." if len(unnamed) > 8 else ""))
    return None, None


def predates_rule(commit: str, enforced_from: str) -> bool:
    """Whether ``commit`` is a strict ancestor of ``enforced_from``. An
    unknown ``enforced_from`` (not in this clone) enforces everything."""

    if not enforced_from or commit.startswith(enforced_from) or enforced_from.startswith(commit):
        return False
    result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, enforced_from],
                            cwd=REPO, capture_output=True)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--enforced-from", default=ENFORCED_FROM,
                        help="commits that are ancestors of this one predate the rule ('' enforces all)")
    args = parser.parse_args()
    ids = delta_rows.row_ids()
    failures, skipped = [], 0
    for commit in git("rev-list", "--reverse", f"{args.base}..{args.head}").split():
        changed = [p for p in git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).split()
                   if p.startswith(SNAPSHOTS)]
        if not changed:
            continue
        if predates_rule(commit, args.enforced_from):
            skipped += 1
            continue
        failure, warning = judge(git("log", "-1", "--format=%B", commit), changed, ids)
        subject = git("log", "-1", "--format=%s", commit).strip()
        if warning:
            print(f"warning: {commit[:12]} {subject}: {warning}")
        if failure:
            failures.append(f"{commit[:12]} {subject}: {failure}")
    if skipped:
        print(f"{skipped} snapshot-changing commit(s) predate the rule ({args.enforced_from[:12]}) and are not checked")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
