"""v2 port of the legacy cta_group::2 accumulating block-scaled GEMM test.

Legacy: tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_cta_group2_batched_gemm_accumulates_both_target_ctas
duplicated the block-scaled ``Tx.gemm_async`` of
``block_scaled_nvfp4_gemm_cta_group2_scale_rows`` with ``accum=True``
(``_append_accumulating_block_scaled_gemm``), transpiled it and checked that
both target CTAs of the cta_group::2 MMA accumulate twice (128 / 256).

The v2 copy keeps the kernel rewrite, the inputs and the exact output
assertion. Dropped: the helper's selection of the block-scaled call by the
``Gemm<BlockScaled<...>>`` variant in the legacy ``emit_rust_module`` text
(legacy ``analyze`` + Rust emission). The kernel has exactly one
``gemm_async`` and it carries ``SFA``/``SFB``, so the port selects it
structurally and asserts that it is the only one.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm import tirx
from tvm_ffi import structural_map

from tests.numsim.support.kernels import block_scaled_nvfp4_gemm_cta_group2_scale_rows
from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
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


@v2_gap(
    "v2 transpile rejects the kernel (UnsupportedTIRxError: site#15/#19 BufferStore "
    "tmem_replicated_view: scale_a_tmem / scale_b_tmem); legacy ran it and produced "
    "128/256. lowering-inventory.md rules direct access to replicated TMEM views "
    "fail-closed (E.3; wave 0 retired the tcgen_cp_* replicated-view tests as out of "
    "scope), but no behaviour-delta row covers it"
)
def test_cta_group2_batched_gemm_accumulates_both_target_ctas():
    """Port of tests/numsim/integration/test_block_scaled_gemm_artifact.py::test_cta_group2_batched_gemm_accumulates_both_target_ctas.

    Dropped: the legacy emitted-Rust variant lookup and ``cache_dir`` (see
    the module docstring).
    """

    accumulated = _append_accumulating_block_scaled_gemm(
        block_scaled_nvfp4_gemm_cta_group2_scale_rows
    )
    left_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    right_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    scale_a = np.full((2, 128, 4), 0x38, dtype=np.uint8)
    scale_b = np.full((2, 256, 4), 0x38, dtype=np.uint8)
    scale_b[:, 128:, :] = np.uint8(0x40)
    output = np.zeros((2, 128, 256), dtype=np.float32)

    module = v2.transpile(accumulated)
    result = v2.Engine().run(
        module,
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    expected = np.full((2, 128, 256), 128.0, dtype=np.float32)
    expected[:, :, 128:] = np.float32(256.0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cta_group2_batched_gemm_replicated_scale_view_fails_closed():
    """Companion of the port above: pins today's v2 behaviour for the same
    kernel. Direct stores to the replicated scale-factor TMEM views fail closed
    at transpile (lowering-inventory.md: "direct access to replicated /
    non-32-bit TMEM views | fails closed by ruling"). Not a legacy assertion;
    delete together with the port if the kernel is retired as out of scope.
    """

    accumulated = _append_accumulating_block_scaled_gemm(
        block_scaled_nvfp4_gemm_cta_group2_scale_rows
    )
    with pytest.raises(UnsupportedTIRxError, match="tmem_replicated_view"):
        v2.transpile(accumulated)
