"""Kernels legacy compiled whose tile ops TVM's dispatch rejects (amended Decision 6).

Each must lower with no ``Unsupported`` once its family in
``v2/lowering/tile_forms/`` is ported. The list is ``legacy_compiled.tsv``
(derived from running legacy ``transpile`` over the sweep residuals); captures
are the serialized PrimFuncs. A family flips from xfail to pass by appearing in
``PORTED``.
"""

from __future__ import annotations

import csv
import pathlib

import pytest

HERE = pathlib.Path(__file__).parent
ROWS = list(csv.DictReader(open(HERE / "legacy_compiled.tsv"), delimiter="\t"))

# Families whose port is complete (owners: gemm/copy_async W1; copy/permute/fill/reduce W12).
PORTED: set[str] = set()
# Individual kernels already lowering (capture hash prefix), before their family is complete.
DONE: set[str] = {
    "3f6092d4", "64fcce42", "40e12a79",  # copy_async: legacy "tma" -> TVM "tma_auto" (W1)
}


@pytest.mark.parametrize("row", ROWS, ids=[f"{r['family']}-{r['kernel']}-{r['capture'][:6]}" for r in ROWS])
def test_legacy_compiled_kernel_lowers(row, request):
    import tvm

    from tirx_harness.numsim.v2.lowering import lower

    if row["family"] not in PORTED and row["capture"][:8] not in DONE:
        request.applymarker(pytest.mark.xfail(reason=f"tile form '{row['family']}' not ported yet", strict=False))
    func = tvm.ir.load_json((HERE / "captures" / f"{row['capture']}.json").read_text())
    program = lower(func, strict=False)
    assert not program.unsupported, program.unsupported[:3]
