from __future__ import annotations

import os

# Many engine processes share the host under `-n 16`; worker-thread
# confinement would stack them on the cores that were idle at launch time.
os.environ["NUMSIM_WORKER_AFFINITY"] = "off"


def pytest_addoption(parser) -> None:
    # Declared at the tests root so every invocation (full run or a targeted
    # one) accepts it. Consumed by tests/conformance/test_conformance.py.
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="rewrite tests/conformance/snapshots from the selected NUMSIM_IMPL instead of comparing",
    )


# ---------------------------------------------------------------------------
# NUMSIM_IMPL=v2: run public-API NumSim tests against the v2 engine.
#
# Rebinds the public names of ``tirx_harness.numsim`` (and the top-level
# ``tirx_harness.racecheck`` / ``synccheck``) to ``tirx_harness.numsim.v2``
# before any test module is imported, so ``from tirx_harness.numsim import
# Engine`` / ``numsim.transpile`` resolve to v2. Internal modules
# (``tirx_harness.numsim.api`` etc.) are untouched; tests importing them keep
# exercising legacy. Data classes shared by both (NumSimCase, TensorMap,
# ComparisonSpec, errors) stay the legacy objects. See dev-loop.md.
if os.environ.get("NUMSIM_IMPL", "").strip() == "v2":
    import tirx_harness as _harness
    import tirx_harness.numsim as _numsim
    from tirx_harness.numsim import v2 as _v2

    for _name in (
        "transpile", "Engine", "compare", "run_case", "CoverageBounds", "ResourceLimits",
        "CompiledModule", "NumSimResult", "ExecutionSubset",
    ):
        setattr(_numsim, _name, getattr(_v2, _name))
    _numsim.NativeAnalysisResult = _v2.AnalysisResult
    _harness.racecheck = _v2.racecheck
    _harness.synccheck = _v2.synccheck
else:
    # The public launch selector is reached as ``tirx_harness.numsim.ExecutionSubset``
    # in both modes (v2 exports it from ``numsim.v2``; legacy only from
    # ``numsim.api``), so tests need no edit when the legacy layer is deleted.
    import tirx_harness.numsim as _numsim
    from tirx_harness.numsim.api import ExecutionSubset as _LegacyExecutionSubset

    if not hasattr(_numsim, "ExecutionSubset"):
        _numsim.ExecutionSubset = _LegacyExecutionSubset
