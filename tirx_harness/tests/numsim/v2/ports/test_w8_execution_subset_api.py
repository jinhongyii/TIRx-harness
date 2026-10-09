"""v2 copy of ``tests/numsim/integration/test_api.py::test_execution_subset_uses_subset_api_name``.

``ExecutionSubset`` is public in v2 (W11-api; docs/api/inputs.md "Launch
selection"): exported from ``tirx_harness.numsim.v2`` / ``.v2.api`` with the
same name, fields and ``to_payload`` shape as the legacy class. The copy pins
the public name and that it selects clusters. The legacy sibling
``test_execution_subset_is_not_publicly_exported`` is the opposite decision
(C: the export is now intended).
"""

import numpy as np

from tirx_harness.numsim import v2
from tirx_harness.numsim.v2 import api
from tirx_harness.numsim.v2.run import Engine


def test_execution_subset_uses_subset_api_name():
    assert v2.ExecutionSubset is api.ExecutionSubset
    assert "ExecutionSubset" in api.__all__ and "ExecutionSubsetSelection" in api.__all__
    subset = v2.ExecutionSubset(cluster_ids=[0, 2])
    assert subset.to_payload() == {"cluster_ids": [0, 2], "cta_ids": None}


def test_execution_subset_selects_clusters():
    from pathlib import Path

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "vector_add.module.json"
    module = v2.from_document(__import__("json").loads(fixture.read_text()))
    assert Engine._subset_extra(v2.ExecutionSubset(cluster_ids=[2, 0]), None, module) == {"subset": (0, 2)}
    inputs = {name: np.zeros(1024, np.float32) for name in ("a", "b", "c")}
    result = v2.Engine().run(module, inputs, subset=v2.ExecutionSubset(cluster_ids=[0]))
    (record,) = [d for d in result.diagnostics if d.get("reason") == "subset_execution"]
    assert record["resident_cluster_ids"] == [0]
