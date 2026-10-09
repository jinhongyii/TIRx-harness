"""Corpus sweep: every canonical and wiki kernel lowers strictly and validates in Rust.

Acceptance check (coordinator): ``Module::from_json`` + ``Program::validate()``
through the engine's own decoder (``numsim_core_py.load_module``), so the check
needs no external tool and survives the legacy deletion. The Rust half is
skipped when the engine extension is not built; the strict lowering half
always runs.
"""

from __future__ import annotations

import warnings

import pytest

from tests.analysis_tools.racecheck.wiki._cases import WIKI_RACECHECK_SPECS, prepare_wiki_racecheck_case
from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES

from tirx_harness.numsim.v2.lowering import lower_module

pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def validator():
    """``numsim_core_py.load_module`` (decode + ``Program::validate``), or None."""
    try:
        from tirx_harness.numsim.v2.compile import native

        return native().load_module
    except Exception:  # noqa: BLE001 - extension not built: lowering half only
        return None


def _check(kernel, validator):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = lower_module(kernel, strict=True)
    assert all(not program.unsupported for program in module.kernels)
    if validator is not None:
        validator(module.to_json())  # raises ValueError on a decode or validate error


@pytest.mark.parametrize("case", CANONICAL_KERNEL_CASES, ids=lambda case: case.name)
def test_canonical_kernel_lowers_and_validates(case, validator):
    _check(case.prepare().kernel, validator)


@pytest.mark.parametrize("spec", WIKI_RACECHECK_SPECS, ids=lambda spec: spec.case_id)
def test_wiki_kernel_lowers_and_validates(spec, validator):
    _check(prepare_wiki_racecheck_case(spec).kernel, validator)
