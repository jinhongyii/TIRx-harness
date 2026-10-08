"""v2 copy of ``tests/numsim/corpus/test_canonical_kernels.py::test_flashmla_small_topk_task_steal_matches_independent_numerical_oracle``.

Legacy ran the canonical ``sparse_flashmla_prefill_head128_small_topk_phase1``
case through ``run_case`` with the legacy-only
``numsim.api.ExecutionSubset(cluster_ids=[0])``. The copy prepares the same
case with the same corpus fixture
(``tests/numsim/corpus/kernels/flashmla.py::prepare_sparse_prefill_case``,
arguments copied from ``canonical_cases._sparse_prefill_head128_small_topk``),
runs ``v2.Engine().run`` with a duck-typed subset whose ``cta_ids`` are exactly
the CTAs of cluster 0 (the second logical cluster stays non-resident and must
be claimed through CLC by the resident pair), and compares every output with
the case's independent NumPy reference and tolerances via ``v2.compare``.
Nothing observable is dropped.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.numsim.corpus.kernels.flashmla import prepare_sparse_prefill_case
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def _static(dims):
    out = []
    for d in dims or (1, 1, 1):
        if isinstance(d, dict):
            d = d["Const"]
        out.append(max(int(d), 1))
    return tuple(out)


def _cluster_zero_cta_ids(module) -> list[int]:
    (kernel,) = module.spec.kernels
    grid = _static(kernel.topology.get("grid"))
    cluster = _static(kernel.topology.get("cluster"))
    return sorted(
        x + y * grid[0] + z * grid[0] * grid[1]
        for x in range(cluster[0])
        for y in range(cluster[1])
        for z in range(cluster[2])
    )


@v2_gap(
    "subset run with only cluster 0 (CTAs 0,1 of a 4-CTA, 2x1x1-cluster launch) resident "
    "completes but never claims the non-resident cluster's task through CLC: query row 1 "
    "of out/max_logits/lse keeps its NaN sentinel (out[1,0,0] nan vs -0.0871); the full "
    "launch matches the oracle. Same failure for the legacy test under NUMSIM_IMPL=v2 and "
    "for subset=cluster_ids=[0] CONTRACT_REQUESTS W12-gaps 6 (scheduler/subset)"
)
def test_flashmla_small_topk_task_steal_matches_independent_numerical_oracle():
    """Port of ``tests/numsim/corpus/test_canonical_kernels.py::test_flashmla_small_topk_task_steal_matches_independent_numerical_oracle``."""

    case = prepare_sparse_prefill_case(
        "sparse_flashmla_prefill_head128_small_topk_phase1",
        s_q=2,
        s_kv=128,
        topk=128,
        d_qk=512,
        h_q=128,
    )
    module = v2.transpile(case.kernel)
    cta_ids = _cluster_zero_cta_ids(module)
    # The launch has more than one cluster, so cluster 0 alone forces a steal.
    grid = _static(module.spec.kernels[0].topology.get("grid"))
    assert len(cta_ids) < grid[0] * grid[1] * grid[2]

    expected = case.reference()
    result = v2.Engine().run(
        module,
        case.args,
        subset=SimpleNamespace(cta_ids=cta_ids, cluster_ids=None),
        assumptions=case.assumptions,
        outputs=case.outputs,
    )
    report = v2.compare(result, expected, tolerances=case.comparisons)

    report.require_ok()
    assert report.verdict == "clean"
    assert report.diagnostics == []
