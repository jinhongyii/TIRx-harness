"""Fold conformance delta snapshots into the base snapshots (step 5).

Usage (repository root)::

    python scripts/numsim-v2/fold_snapshot_deltas.py            # dry run
    python scripts/numsim-v2/fold_snapshot_deltas.py --apply    # rewrite files
    python scripts/numsim-v2/fold_snapshot_deltas.py --root /tmp/copy/snapshots --apply

While the legacy engine exists, ``<case>/<mode>.json`` is the legacy oracle
and ``<case>/<mode>.delta.json`` holds the v2-expected result for a case where
a behaviour-delta row rules legacy wrong. When the legacy engine is deleted,
v2 becomes the oracle: each delta file replaces its base snapshot (minus its
``delta`` field) and is removed.

The run prints a Markdown table (case, mode, cited delta rows) and the
``Snapshot-Regen:`` trailer for the deletion commit's message, so
``check_snapshot_deltas.py`` accepts the commit; row ids are printed
file-qualified (``racecheck B7``). Exits 1 when a delta file cites no known
row, or a bare id that several tables share (``delta_rows.py``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "tirx_harness/tests/conformance/snapshots"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta_rows  # noqa: E402


def dumps(snapshot: dict) -> str:
    # Same serialization as tests/conformance/snapshot.py::dumps.
    return json.dumps(snapshot, indent=1, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="snapshot directory")
    parser.add_argument("--apply", action="store_true", help="rewrite the files (default: dry run)")
    args = parser.parse_args()

    ids = delta_rows.row_ids()
    folded, unjustified = [], []
    for delta_path in sorted(args.root.glob("*/*.delta.json")):
        case = delta_path.parent.name
        mode = delta_path.name[: -len(".delta.json")]
        data = json.loads(delta_path.read_text())
        reason = str(data.pop("delta", ""))
        citations = delta_rows.parse(reason, ids)
        cited = sorted(citations.qualified)
        if not citations.ok:
            why = reason if not citations.ambiguous else f"ambiguous bare id(s) {sorted(citations.ambiguous)}: {reason}"
            unjustified.append((case, mode, why))
            continue
        base = delta_path.with_name(f"{mode}.json")
        if not base.exists():
            unjustified.append((case, mode, "no base snapshot"))
            continue
        folded.append((case, mode, cited))
        if args.apply:
            base.write_text(dumps(data))
            delta_path.unlink()

    print("| case | mode | delta rows |")
    print("| --- | --- | --- |")
    for case, mode, cited in folded:
        print(f"| `{case}` | {mode} | {', '.join(cited)} |")
    all_rows = sorted({r for _, _, cited in folded for r in cited})
    print()
    print(f"Snapshot-Regen: schema fold {len(folded)} delta snapshots into the v2 oracle "
          f"(rows {', '.join(all_rows)})")
    if not args.apply:
        print("\n(dry run: nothing written; pass --apply)")
    for case, mode, reason in unjustified:
        print(f"error: {case}/{mode}.delta.json: no known delta row ({reason[:80]!r})", file=sys.stderr)
    return 1 if unjustified else 0


if __name__ == "__main__":
    sys.exit(main())
