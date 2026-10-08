"""Behaviour-delta row ids and their citations (shared by the snapshot tools).

Row ids repeat across the three delta tables (``T4`` is a row of all three),
so a citation names its table (coordinator decision, 2026-10-08):

* qualified: ``numsim T4``, ``racecheck B7``, ``sync M14``, or the table's
  file name (``racecheck-behaviour-deltas R3``, optionally with ``.md``). A
  qualifier covers the ids listed right after it, joined by ``,``, ``+``,
  ``/`` or ``and`` (``racecheck R3 + T18``);
* bare: ``B7`` is accepted only when exactly one table has that row; a bare
  id present in several tables is *ambiguous* and must be qualified.

No renumbering: the ids stay as the tables define them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLES = {
    "numsim": REPO / "docs/development/numsim-behaviour-deltas.md",
    "racecheck": REPO / "docs/development/racecheck-behaviour-deltas.md",
    "sync": REPO / "docs/development/sync-behaviour-deltas.md",
}
_ID = r"[A-Z]{1,2}\d{1,3}"
ROW_ID = re.compile(rf"(?<!ISA )\b({_ID})\b(?!-\d)")
QUALIFIED = re.compile(
    rf"\b(numsim|racecheck|sync)(?:-behaviou?r-deltas(?:\.md)?)?\s*:?\s+"
    rf"((?<!ISA ){_ID}\b(?!-\d)(?:\s*(?:,|\+|/|\band\b)\s*{_ID}\b(?!-\d))*)"
)


def row_ids(tables: dict[str, Path] = TABLES) -> dict[str, set[str]]:
    out = {}
    for table, path in tables.items():
        ids = set()
        for line in path.read_text().splitlines():
            match = re.match(rf"\|\s*({_ID})\s*\|", line)
            if match:
                ids.add(match.group(1))
        out[table] = ids
    return out


@dataclass
class Citations:
    qualified: set[str] = field(default_factory=set)  # "racecheck B7"
    ambiguous: set[str] = field(default_factory=set)  # bare ids in several tables
    unknown: set[str] = field(default_factory=set)  # qualified ids the table lacks

    @property
    def ok(self) -> bool:
        return bool(self.qualified) and not self.ambiguous and not self.unknown


def parse(text: str, ids: dict[str, set[str]]) -> Citations:
    found = Citations()
    covered: list[tuple[int, int]] = []
    for match in QUALIFIED.finditer(text):
        table = match.group(1)
        covered.append(match.span(2))
        for row in ROW_ID.findall(match.group(2)):
            (found.qualified if row in ids[table] else found.unknown).add(f"{table} {row}")
    for match in ROW_ID.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        row = match.group(1)
        tables = [table for table, rows in ids.items() if row in rows]
        if len(tables) == 1:
            found.qualified.add(f"{tables[0]} {row}")
        elif len(tables) > 1:
            found.ambiguous.add(row)
    return found


def qualify(text: str, ids: dict[str, set[str]], table_hint: str | None = None) -> str:
    """Rewrite bare row ids as qualified ones: unique ids get their table;
    ambiguous ids get ``table_hint`` when that table has the row."""

    covered = [m.span(2) for m in QUALIFIED.finditer(text)]
    out, last = [], 0
    for match in ROW_ID.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        row = match.group(1)
        tables = [table for table, rows in ids.items() if row in rows]
        if len(tables) == 1:
            table = tables[0]
        elif table_hint in tables:
            table = table_hint
        else:
            continue
        out.append(text[last:match.start()] + f"{table} {row}")
        last = match.end()
    out.append(text[last:])
    return "".join(out)
