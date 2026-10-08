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
    if update and implementation.name != snap.oracle_impl_name():
        pytest.fail(
            f"snapshots are the {snap.oracle_impl_name()} oracle; regenerate them only from it "
            "(while the legacy engine exists: NUMSIM_IMPL=legacy)"
        )
    expected = (
        snap.load_snapshot(entry.name, mode) if update
        else snap.load_expected(entry.name, mode, implementation.name)
    )
    if not update and implementation.name == "v2" and expected is not None and "error" in expected:
        # Legacy could not run this case; it gets a v2 snapshot under the
        # post-deletion policy once its open items are cleared.
        pytest.skip(f"legacy failed this case ({expected['error']}); there is no oracle to compare")

    if implementation.name == "legacy":
        actual = snap.collect_snapshot(entry, mode, implementation)
    else:
        try:
            actual = snap.collect_snapshot(entry, mode, implementation, reraise=(NotImplementedError,))
        except NotImplementedError as error:
            pytest.skip(f"{implementation.name}: not implemented yet: {error}")

    if update:
        snap.write_snapshot(entry.name, mode, actual)
        return
    if expected is None:
        pytest.fail(
            f"no snapshot for {entry.name}/{mode}; generate it with "
            "NUMSIM_IMPL=legacy pytest tests/conformance --update-snapshots"
        )
    if implementation.name != snap.oracle_impl_name():
        actual = snap.relax_unanchored(expected, actual)
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
    # Every delta snapshot names its behaviour-delta row.
    for path in snap.SNAPSHOT_ROOT.glob("*/*.delta.json"):
        assert snap.json.loads(path.read_text()).get("delta"), path


def test_post_deletion_mode_selects_v2(monkeypatch, tmp_path):
    """Once the legacy package is gone: v2 is the oracle, NUMSIM_IMPL no
    longer selects legacy, and delta files are not consulted (they were
    folded into the base snapshots)."""

    monkeypatch.setattr(snap, "legacy_available", lambda: False)
    monkeypatch.delenv(snap.IMPL_ENV, raising=False)
    assert snap.selected_impl_name() == "v2"
    assert snap.oracle_impl_name() == "v2"
    monkeypatch.setenv(snap.IMPL_ENV, "legacy")
    with pytest.raises(ValueError, match="deleted"):
        snap.selected_impl_name()
    case = tmp_path / "case"
    case.mkdir()
    (case / "numsim.json").write_text(snap.dumps({"schema": 4, "case": "case", "mode": "numsim", "verdict": "clean"}))
    (case / "numsim.delta.json").write_text(snap.dumps({"delta": "X4", "verdict": "review"}))
    assert snap.load_expected("case", "numsim", "v2", root=tmp_path)["verdict"] == "clean"
