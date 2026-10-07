"""Corpus conformance: every canonical case x {numsim, racecheck, synccheck}.

Compares the selected implementation (``NUMSIM_IMPL=legacy|v2``, default
``legacy``) with the frozen snapshots in ``snapshots/``. Pass
``--update-snapshots`` to rewrite them instead. See README.md.
"""

from __future__ import annotations

import pytest

from tests.conformance import snapshot as snap
from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES


@pytest.fixture(scope="session")
def implementation():
    try:
        return snap.load_implementation()
    except snap.ImplementationUnavailable as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("mode", snap.MODES)
@pytest.mark.parametrize("entry", CANONICAL_KERNEL_CASES, ids=lambda entry: entry.name)
def test_conformance_snapshot(entry, mode, implementation, request) -> None:
    update = request.config.getoption("update_snapshots")
    expected = snap.load_snapshot(entry.name, mode)
    if not update and implementation.name != "legacy" and expected is not None and "error" in expected:
        pytest.skip(f"legacy failed this case ({expected['error']}); there is no oracle to compare")

    actual = snap.collect_snapshot(entry, mode, implementation)

    if update:
        snap.write_snapshot(entry.name, mode, actual)
        return
    if expected is None:
        pytest.fail(
            f"no snapshot for {entry.name}/{mode}; generate it with "
            "NUMSIM_IMPL=legacy pytest tests/conformance --update-snapshots"
        )
    if actual != expected:
        pytest.fail(
            f"{implementation.name} diverges from the {entry.name}/{mode} snapshot:\n"
            + snap.diff_snapshots(expected, actual)[:20000],
            pytrace=False,
        )


def test_snapshot_directory_matches_corpus() -> None:
    """Snapshots exist for exactly the canonical cases (no stale leftovers)."""

    names = {entry.name for entry in CANONICAL_KERNEL_CASES}
    present = {path.name for path in snap.SNAPSHOT_ROOT.iterdir() if path.is_dir()} if snap.SNAPSHOT_ROOT.exists() else set()
    assert present <= names, sorted(present - names)
