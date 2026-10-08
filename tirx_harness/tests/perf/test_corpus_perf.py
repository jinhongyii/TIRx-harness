"""NumSim v2 corpus performance gate: relative baselines on real kernels.

Two groups of workloads, run on the interpreter (the only v2 executor):

* The Mega MoE workloads of the legacy performance gates: the NumSim maximum
  configuration and the six Racecheck configurations on the 148-SM grid.
  Metric ``v2.interp.<mode>.mega_moe.<workload>`` is the ``Engine.run`` /
  ``run_racecheck_phase`` wall time. Preparation and lowering are not timed.
* Canonical conformance cases that guard checker rules on real kernels
  (W6/W10: the synccheck explorer regression on the dglu_dbias GEMMs and
  flash_attention4, and three racecheck cases). Each case runs at 1 and 32
  engine workers. It records the engine ``run`` phase and, for synccheck, the
  explorer's ``check`` phase separately (``payload["timing"]``), as
  ``v2.interp.<mode>.<case>.<phase>.w<workers>``. Racecheck checks inside
  the run, so its rule cost is in ``run``.

Each test attaches its measurements to the junit report (``perf_metrics``,
a JSON object of metric -> seconds) before asserting correctness. With
``NUMSIM_PERF_ENFORCE=0`` it records without comparing.
``scripts/numsim-v2/perf_gate.py`` runs the module that way and applies the
load policy. Otherwise every metric is compared with
``baselines/v2/<host-class>.json`` (``perf_baseline``, suite ``v2``).
"""

from __future__ import annotations

import copy
import json
import os
from time import perf_counter

import pytest

from tests.conformance import snapshot as snap
from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES
from tests.numsim.corpus.kernels.deepgemm import prepare_mega_moe_case
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tests.perf.perf_baseline import assert_within_baseline
from tirx_harness.numsim import v2

pytestmark = pytest.mark.performance

SUITE = "v2"
BACKEND = "interp"  # metric-name prefix; the engine has one executor
WORKERS = 16
LOOP_BUDGET = 10_000_000

# The legacy gates' workloads (tests/numsim/corpus/test_canonical_kernels.py,
# tests/analysis_tools/racecheck/corpus/test_canonical_kernels_racecheck.py).
MEGA_MOE_MAX_CONFIG = "t8192_m8192_h7168_i3072_e384_k6_g1"
RACECHECK_NUM_SMS = 148
RACECHECK_MEDIUM = "t64_h2048_i1536_e96_k4_g1"
RACECHECK_CASES = {
    "two_tokens": "p1_tok2_h1024_i512_e2_k1_bm16",
    "sixteen_tokens": "p1_tok16_h1024_i512_e2_k2_bm32",
    "twenty_four_experts": "t8_h1024_i512_e24_k2_g1",
    "shared_expert": "p1_tok2_h1024_i512_e2_k1_bm16_s1",
    "medium_moe": RACECHECK_MEDIUM,
    "large_moe": "t64_h4096_i1536_e96_k4_g1",
}
# v2 Mega MoE racecheck kinds on the 148-SM grid (W5): findings B1/R4 and B7
# `scope_mismatch`, T19 `data_race`; advisories X4 and P7. Verdict `error`.
V2_FINDING_KINDS = {"scope_mismatch", "data_race"}
V2_ADVISORY_KINDS = {"cross_cta_async_order", "alias_stale_read"}

# Canonical cases whose checker phase guards a rule on a real kernel.
CHECKER_CASES = [
    ("synccheck", "cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias"),
    ("synccheck", "cudnn_sm100_moe_grouped_gemm_dglu_dbias"),
    ("synccheck", "flash_attention4"),
    ("racecheck", "recurrent_kda_decode_one_warp"),
    ("racecheck", "gdn_decode_bf16_wide_vec_mtp"),
    ("racecheck", "selective_state_update_stp_simple"),
]
CHECKER_WORKERS = (1, 32)


def _enforce() -> bool:
    return os.environ.get("NUMSIM_PERF_ENFORCE", "1") != "0"


def _record(record_property, metrics: dict[str, float]) -> None:
    """Attach the measurements before the correctness asserts, so a gate run
    reports the time of a run whose verdict also regressed."""

    record_property("perf_metrics", json.dumps({k: round(v, 4) for k, v in metrics.items()}))


def _enforce_baselines(metrics: dict[str, float]) -> None:
    if _enforce():
        for metric, seconds in metrics.items():
            assert_within_baseline(metric, seconds, suite=SUITE)


def _mega_moe_configs():
    module = load_tirx_kernel("sm100_fp8_fp4_mega_moe")
    configs = {config["label"]: config for config in module.CONFIGS}
    # As the legacy gate: the medium case halves this shape's hidden width.
    configs[RACECHECK_MEDIUM] = {**configs["t64_h4096_i1536_e96_k4_g1"], "hidden": 2048, "label": RACECHECK_MEDIUM}
    return configs


def _engine(workers: int) -> v2.Engine:
    return v2.Engine(max_workers=workers, native_loop_iteration_budget=LOOP_BUDGET)


def test_mega_moe_numsim_max_config(record_property) -> None:
    case = prepare_mega_moe_case(_mega_moe_configs()[MEGA_MOE_MAX_CONFIG])
    module = v2.transpile(case.kernel)
    expected = copy.deepcopy(case.reference())
    started = perf_counter()
    result = _engine(WORKERS).run(
        module, case.args, subset=case.subset, assumptions=case.assumptions, outputs=case.outputs
    )
    metrics = {f"v2.{BACKEND}.numsim.mega_moe.max_config": perf_counter() - started}
    _record(record_property, metrics)
    v2.compare(result, expected, tolerances=case.comparisons).require_ok()
    assert result.verdict == "clean", result.diagnostics
    _enforce_baselines(metrics)


@pytest.mark.parametrize("workload", list(RACECHECK_CASES))
def test_mega_moe_racecheck(workload: str, record_property, monkeypatch) -> None:
    monkeypatch.setenv("TIRX_DEEPGEMM_NUM_SMS_OVERRIDE", str(RACECHECK_NUM_SMS))
    case = prepare_mega_moe_case(_mega_moe_configs()[RACECHECK_CASES[workload]])
    module = v2.transpile(case.kernel)
    started = perf_counter()
    result = _engine(WORKERS).run_racecheck_phase(module, case.args, phase_index=0, subset=case.subset)
    metrics = {f"v2.{BACKEND}.racecheck.mega_moe.{workload}": perf_counter() - started}
    _record(record_property, metrics)
    payload = result.to_dict()
    assert result.verdict == "error", payload
    assert {item["kind"] for item in result.findings} <= V2_FINDING_KINDS, payload
    assert {item["kind"] for item in result.advisories} <= V2_ADVISORY_KINDS, payload
    assert payload["incomplete"] == [] and payload["execution_error"] is None, payload
    _enforce_baselines(metrics)


@pytest.mark.parametrize("workers", CHECKER_WORKERS, ids=lambda w: f"w{w}")
@pytest.mark.parametrize(("mode", "name"), CHECKER_CASES, ids=[f"{m}-{n}" for m, n in CHECKER_CASES])
def test_checker_phase_on_canonical_case(mode: str, name: str, workers: int, record_property) -> None:
    entry = next(item for item in CANONICAL_KERNEL_CASES if item.name == name)
    case = entry.prepare()
    module = v2.transpile(case.kernel)
    engine = _engine(workers)  # fresh engine: no memoized run
    if mode == "racecheck":
        result = engine.run_racecheck_phase(module, case.args, phase_index=0, subset=case.subset)
    else:
        result = engine.run_synccheck_phase(module, case.args, phase_index=0, subset=case.subset)
    payload = result.to_dict()
    timing = payload["timing"]
    prefix = f"v2.{BACKEND}.{mode}.{name}"
    metrics = {f"{prefix}.run.w{workers}": timing["run"] / 1e3}
    if mode == "synccheck":
        # Racecheck checks inline (its observer runs inside the engine run),
        # so its `check` phase is ~0 and `run` carries the rule cost.
        metrics[f"{prefix}.check.w{workers}"] = timing["check"] / 1e3
    _record(record_property, metrics)
    assert payload["execution_error"] is None, payload
    expected = snap.load_expected(name, mode, "v2")
    if expected and expected.get("phases"):
        assert result.verdict == expected["phases"][0]["verdict"], payload
    _enforce_baselines(metrics)
