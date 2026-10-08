"""``numsim-core/src/oplib/lowering_ops.tsv`` (input to the generated
``numsim-oplib/SUPPORTED_OPS.md``) matches the v2 lowering tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/numsim-v2/lowering_ops.py"


def test_lowering_ops_inventory_is_current():
    spec = importlib.util.spec_from_file_location("lowering_ops", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUT.read_text() == module.render(), (
        "lowering_ops.tsv is stale: run scripts/numsim-v2/lowering_ops.py, then "
        "cargo run -p numsim-core --example supported_ops"
    )
