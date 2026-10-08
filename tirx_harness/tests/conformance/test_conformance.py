"""Corpus conformance: every canonical case x {numsim, racecheck, synccheck}.

Compares ``tirx_harness.numsim`` with the frozen snapshots in ``snapshots/``.
Pass ``--update-snapshots`` to rewrite them instead; a commit that changes a
snapshot cites the behaviour-delta row that justifies it (README.md).
"""

from __future__ import annotations

import pytest

from tests.conformance import snapshot as snap
from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES


@pytest.mark.parametrize("mode", snap.MODES)
@pytest.mark.parametrize("entry", CANONICAL_KERNEL_CASES, ids=lambda entry: entry.name)
def test_conformance_snapshot(entry, mode, request) -> None:
    actual = snap.collect_snapshot(entry, mode)
    if request.config.getoption("update_snapshots"):
        snap.write_snapshot(entry.name, mode, actual)
        return
    expected = snap.load_snapshot(entry.name, mode)
    if expected is None:
        pytest.fail(
            f"no snapshot for {entry.name}/{mode}; generate it with "
            "pytest tests/conformance --update-snapshots"
        )
    if actual != expected:
        pytest.fail(
            f"NumSim diverges from the {entry.name}/{mode} snapshot:\n"
            + snap.diff_snapshots(expected, actual)[:20000],
            pytrace=False,
        )


def test_snapshot_directory_matches_corpus() -> None:
    """Snapshots exist for exactly the canonical cases (no stale leftovers),
    and no delta file survives the step-5 fold."""

    names = {entry.name for entry in CANONICAL_KERNEL_CASES}
    present = {path.name for path in snap.SNAPSHOT_ROOT.iterdir() if path.is_dir()} if snap.SNAPSHOT_ROOT.exists() else set()
    assert present <= names, sorted(present - names)
    assert not list(snap.SNAPSHOT_ROOT.glob("*/*.delta.json"))
