"""Corpus sweep: every canonical and wiki kernel lowers strictly and validates in Rust.

Acceptance check (coordinator): ``numsim_core::program::Module::from_json`` +
``Program::validate()`` + postcard round trip, via
``core-rs/tools/validate-program`` built by ``scripts/numsim-v2/validate.sh``.
The Rust half is skipped when cargo is unavailable; the strict lowering half
always runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from tests.analysis_tools.racecheck.wiki._cases import WIKI_RACECHECK_SPECS, prepare_wiki_racecheck_case
from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES

from tirx_harness.numsim.v2.lowering import lower_module

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[4]
_VALIDATE = _REPO / "scripts" / "numsim-v2" / "validate.sh"


@pytest.fixture(scope="session")
def validator(tmp_path_factory):
    if shutil.which("cargo") is None:
        return None
    work = tmp_path_factory.mktemp("validate-program")
    env = dict(os.environ, VALIDATE_WORK=str(work))
    subprocess.run([str(_VALIDATE)], check=True, env=env, capture_output=True, text=True)
    return work / "target" / "release" / "validate-program"


def _check(kernel, tmp_path, validator):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = lower_module(kernel, strict=True)
    assert all(not program.unsupported for program in module.kernels)
    if validator is None:
        return
    path = tmp_path / "module.json"
    path.write_text(module.to_json())
    result = subprocess.run([str(validator), str(path)], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.startswith("OK"), result.stdout + result.stderr


@pytest.mark.parametrize("case", CANONICAL_KERNEL_CASES, ids=lambda case: case.name)
def test_canonical_kernel_lowers_and_validates(case, tmp_path, validator):
    _check(case.prepare().kernel, tmp_path, validator)


@pytest.mark.parametrize("spec", WIKI_RACECHECK_SPECS, ids=lambda spec: spec.case_id)
def test_wiki_kernel_lowers_and_validates(spec, tmp_path, validator):
    _check(prepare_wiki_racecheck_case(spec).kernel, tmp_path, validator)
