"""``scripts/numsim-v2/check_snapshot_deltas.py``: the CI rule that a commit
changing a conformance snapshot cites a file-qualified behaviour-delta row
(or carries a ``Snapshot-Regen: schema`` trailer), enforced only from the
commit that introduced the rule on."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts" / "numsim-v2"


@pytest.fixture(scope="module")
def check():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("check_snapshot_deltas", SCRIPTS / "check_snapshot_deltas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IDS = {"numsim": {"H5", "T4"}, "racecheck": {"B7", "T4", "X4"}, "sync": {"S1", "T4"}}
SNAP = "tirx_harness/tests/conformance/snapshots/"


def test_judge_rules(check):
    changed = [SNAP + "bmm_fp8_rubin/racecheck.json"]
    assert check.judge("fix: bmm_fp8_rubin racecheck B7", changed, IDS) == (None, None)
    assert check.judge("fix: bmm_fp8_rubin (X4)", changed, IDS) == (None, None)  # unique bare id
    failure, _ = check.judge("fix: bmm_fp8_rubin T4", changed, IDS)  # T4 is in all three tables
    assert "ambiguous" in failure and "T4" in failure
    failure, _ = check.judge("fix: bmm_fp8_rubin racecheck S1", changed, IDS)
    assert "unknown" in failure and "racecheck S1" in failure
    failure, _ = check.judge("refresh bmm_fp8_rubin", changed, IDS)
    assert "cites no delta row id" in failure
    assert check.judge("regen\n\nSnapshot-Regen: schema projection", changed, IDS) == (None, None)
    assert check.judge("regen\n\nSnapshot-Regen: harness inputs without BLAS", changed, IDS) == (None, None)
    assert check.judge("regen\n\nSnapshot-Regen: numerics libm", changed, IDS) != (None, None)
    failure, warning = check.judge("racecheck B7 for the gemm", changed, IDS)
    assert failure is None and "bmm_fp8_rubin" in warning


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_commits_before_the_rule_are_skipped(check, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    snap = repo / SNAP / "case"
    snap.mkdir(parents=True)

    def commit(message: str) -> str:
        (snap / "numsim.json").write_text(message)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", message)
        return _git(repo, "rev-parse", "HEAD")

    (repo / "README").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    old = commit("initial snapshots (no row: predates the rule)")
    rule = commit("introduce the rule: case racecheck B7")
    new = commit("unjustified change to case")
    monkeypatch.setattr(check, "REPO", repo)
    assert check.predates_rule(old, rule)
    assert not check.predates_rule(rule, rule)  # the rule commit itself is checked
    assert not check.predates_rule(new, rule)
    assert not check.predates_rule(old, "")  # '' enforces everything
    assert not check.predates_rule(old, "0" * 40)  # unknown cutoff: enforce

    monkeypatch.setattr(check.delta_rows, "row_ids", lambda *a, **k: IDS)
    # base..rule contains the unjustified pre-rule commit: skipped, passes.
    monkeypatch.setattr(sys, "argv", ["check", "--base", base, "--head", rule, "--enforced-from", rule])
    assert check.main() == 0
    # Enforcing everything fails on it.
    monkeypatch.setattr(sys, "argv", ["check", "--base", base, "--head", rule, "--enforced-from", ""])
    assert check.main() == 1
    # A post-rule unjustified commit fails.
    monkeypatch.setattr(sys, "argv", ["check", "--base", base, "--head", new, "--enforced-from", rule])
    assert check.main() == 1
