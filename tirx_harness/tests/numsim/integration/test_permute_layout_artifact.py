from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.support.kernels import (
    global_permute_layout_roundtrip,
    permuted_global_layout_read,
    shared_permute_layout_roundtrip,
    shared_permute_layout_zero_fills_bf16_padding,
    shared_permute_layout_zero_fills_fp8_padding,
    shared_permute_layout_zero_fills_fp16_padding,
    shared_permute_layout_zero_fills_uninitialized_padding,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout

_EXPLICIT_PERMUTED_LAYOUT = TileLayout(S[(4, 32) : (1, 4)])


@T.prim_func
def _explicit_permute_layout_dispatch(
    source: T.Buffer((128,), "uint32"), output: T.Buffer((128,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_shared = T.alloc_buffer((128,), "uint32", scope="shared")
    permuted_shared = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_EXPLICIT_PERMUTED_LAYOUT
    )
    if lane == 0:
        Tx.copy(source_shared[:], source[:])
    T.cuda.warp_sync()
    Tx.warp.permute_layout(permuted_shared[:], source_shared[:], dispatch="warp_xor_swizzle")
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy(output[:], permuted_shared[:])


def test_global_nondefault_layout_consumes_physical_host_order(tmp_path):
    logical = np.arange(128, dtype=np.uint32) * np.uint32(13) + np.uint32(5)
    physical = np.empty_like(logical)
    logical_index = np.arange(128)
    physical_index = (logical_index // 32) + (logical_index % 32) * 4
    physical[physical_index] = logical
    output = np.zeros_like(logical)

    module = numsim.transpile(permuted_global_layout_read, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": physical, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], logical)
