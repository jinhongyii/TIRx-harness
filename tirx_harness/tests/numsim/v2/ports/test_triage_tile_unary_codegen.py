"""v2 delta copy of ``tests/numsim/runtime/test_tile_unary_codegen.py::test_warpgroup_shared_owner_is_independent_of_vector_chunk``.

Delta: Decision 6, "tile semantics are TVM's dispatch output"
(``docs/development/lowering-inventory.md`` D.2; ruling quoted in
``docs/development/numsim-behaviour-deltas.md``, "Tile reductions ... Decision 6").
The scalar operand ``T.cast(thread, "float32")`` is evaluated by whichever thread
TVM's ``dispatch="smem"`` code assigns to each element. Legacy used its own
element map (owner of element ``i`` = ``i % 128`` for every op, "independent of
vector chunk"). The dispatched code chunks contiguously instead:

- ``Tx.wg.add`` (f32x2 packed, ``add.rz.ftz.f32x2``): thread ``t`` owns elements
  ``2t, 2t+1`` and ``256+2t, 256+2t+1``, i.e. owner = ``(i // 2) % 128``, and the
  sum is rounded toward zero;
- ``Tx.wg.fdiv`` (vector of 4): thread ``t`` owns ``4t .. 4t+3``, owner = ``i // 4``.

On hardware the dispatched code is what runs, so these are the values a GPU gives.
Kept: the shared-memory tile op applies the per-thread scalar to every element
exactly once, with the dispatch's owner.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tests.numsim.v2.ports._triage_rz import add_rz_ftz_f32
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def tile_warpgroup_shared_vector_owners(
    source: T.Buffer((512,), "float32"), output: T.Buffer((2, 512), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    packed = T.alloc_buffer((512,), "float32", scope="shared", layout=TileLayout(S[512]))
    scalar = T.alloc_buffer((512,), "float32", scope="shared", layout=TileLayout(S[512]))
    for index in T.serial(4):
        linear = thread + index * 128
        packed[linear] = source[linear]
        scalar[linear] = source[linear]
    T.cuda.cta_sync()

    Tx.wg.add(packed, packed, T.cast(thread, "float32"), dispatch="smem")
    Tx.wg.fdiv(
        scalar,
        scalar,
        T.cast(thread + 1, "float32"),
        dispatch="smem",
    )
    for index in T.serial(4):
        linear = thread + index * 128
        output[0, linear] = packed[linear]
        output[1, linear] = scalar[linear]


def test_warpgroup_shared_owner_is_independent_of_vector_chunk():
    """Delta copy (Decision 6: dispatched owner map); see the module docstring."""
    source = np.linspace(1.0, 2.0, 512, dtype=np.float32)
    output = np.zeros((2, 512), dtype=np.float32)

    result = v2.Engine().run(
        v2.transpile(tile_warpgroup_shared_vector_owners), {"source": source, "output": output}
    )

    indices = np.arange(512, dtype=np.int64)
    add_owner = ((indices // 2) % 128).astype(np.float32)
    div_owner = (indices // 4).astype(np.float32)
    np.testing.assert_array_equal(result.outputs["output"][0], add_rz_ftz_f32(source, add_owner))
    np.testing.assert_allclose(
        result.outputs["output"][1], source / (div_owner + np.float32(1)), rtol=2e-6, atol=2e-6
    )
