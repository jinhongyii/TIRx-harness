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
PORTED: set[str] = {"copy", "fill", "permute", "reduce"}
# Individual kernels already lowering (capture hash prefix), before their family is complete.
DONE: set[str] = {
    "3f6092d4", "64fcce42", "40e12a79",  # copy_async: legacy "tma" -> TVM "tma_auto" (W1)
    "c9c5f890",  # gemm_async nested_elect_warp_gemm: implicit single warpgroup declared (W1)
}

# Kernels of a ported family still blocked by a rule outside the tile form.
BLOCKED: dict[str, str] = {
    # permute_layout lowers; the kernel then reads a replicated TMEM view directly
    # (numsim-behaviour-deltas L1, contract item 29).
    "afa1d772": "tmem_replicated_view (delta L1)",
}

# Forms legacy accepted but the hardware rejects (ruling 2026-10-08,
# numsim-isa-answers.md; delta rows L4-L7): v2 must fail closed on them. The
# public tests move to valid shapes (W9).
HW_INVALID: dict[str, str] = {
    **dict.fromkeys(["0e9b7884", "14f16144", "4d6f137d", "77e30a85", "810f07d0", "e43b7b8c"],
                    "L6 TMA inner box 8 bytes"),
    "8960cb4a": "L7 TMA into a padded shared slice (phase change)",
    **dict.fromkeys(["7c87de29", "6a205528", "2bfa4404"], "L5 block-scaled SFA K extent 8"),
    **dict.fromkeys(["354eea5a", "016eda97", "e4d8a4a4", "ade818c4", "ae6de81e", "f5034afe", "5501e406",
                     "b0a8bc90", "067520aa", "5aed5aa2", "8d2db317", "6e8be779", "7e3d1cc8", "d024eb22"],
                    "L4 invalid tcgen05.mma shape"),
}


@pytest.mark.parametrize("row", ROWS, ids=[f"{r['family']}-{r['kernel']}-{r['capture'][:6]}" for r in ROWS])
def test_legacy_compiled_kernel_lowers(row, request):
    import tvm

    from tirx_harness.numsim.v2.lowering import lower

    if row["capture"][:8] in HW_INVALID:
        func = tvm.ir.load_json((HERE / "captures" / f"{row['capture']}.json").read_text())
        program = lower(func, strict=False)
        assert program.unsupported, f"{HW_INVALID[row['capture'][:8]]}: must fail closed"
        return
    if row["family"] not in PORTED and row["capture"][:8] not in DONE:
        request.applymarker(pytest.mark.xfail(reason=f"tile form '{row['family']}' not ported yet", strict=False))
    if row["capture"][:8] in BLOCKED:
        request.applymarker(pytest.mark.xfail(reason=BLOCKED[row["capture"][:8]], strict=True))
    func = tvm.ir.load_json((HERE / "captures" / f"{row['capture']}.json").read_text())
    program = lower(func, strict=False)
    assert not program.unsupported, program.unsupported[:3]
