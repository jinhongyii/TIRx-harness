"""v2 port of the legacy cta_group::2 accumulating block-scaled GEMM test.

Legacy: tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_cta_group2_batched_gemm_accumulates_both_target_ctas
duplicated the block-scaled ``Tx.gemm_async`` of
``block_scaled_nvfp4_gemm_cta_group2_scale_rows`` with ``accum=True``
(``_append_accumulating_block_scaled_gemm``), transpiled it and checked that
both target CTAs of the cta_group::2 MMA accumulate twice (128 / 256).

The v2 copy is an expected-outcome test (numsim-behaviour-deltas L1): the
kernel accesses replicated scale-factor TMEM views directly, which v2 rejects at
transpile. Dropped: the helper's selection of the block-scaled call by the
``Gemm<BlockScaled<...>>`` variant in the legacy ``emit_rust_module`` text; the
port selects the one ``gemm_async`` carrying ``SFA``/``SFB`` structurally.
"""

from __future__ import annotations

import pytest
from tvm import tirx
from tvm_ffi import structural_map

from tests.numsim.support.kernels import block_scaled_nvfp4_gemm_cta_group2_scale_rows
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


def _append_accumulating_block_scaled_gemm(func):
    """Legacy helper without the emitted-Rust variant lookup: follow the
    single ``gemm_async`` with a copy whose trailing ``accum`` is True."""

    seen = []

    def rewrite(node):
        if type(node).__name__ != "TilePrimitiveCall":
            return node
        if str(node.op.name) != "tirx.tile.gemm_async":
            return node
        seen.append(node)
        arguments = (*node.args[:-1], True)
        return tirx.SeqStmt([node, node.replace(args=arguments)])

    body = structural_map(func.body, (tirx.TilePrimitiveCall, rewrite))
    assert len(seen) == 1
    return func.with_body(body)


def test_cta_group2_batched_gemm_accumulates_both_target_ctas():
    """Port of tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_cta_group2_batched_gemm_accumulates_both_target_ctas,
    as an expected-outcome test (W11).

    numsim-behaviour-deltas L1 (contract item 29): the kernel stores to the
    replicated scale-factor TMEM views ``scale_a_tmem`` / ``scale_b_tmem``
    directly, which v2 rejects at transpile with ``tmem_replicated_view``.
    Legacy ran it (128 / 256 per target CTA); that outcome has no v2
    counterpart by ruling. The kernel rewrite is kept so the rejected kernel is
    exactly the legacy one.
    """

    accumulated = _append_accumulating_block_scaled_gemm(
        block_scaled_nvfp4_gemm_cta_group2_scale_rows
    )
    with pytest.raises(UnsupportedTIRxError, match="tmem_replicated_view"):
        v2.transpile(accumulated)
