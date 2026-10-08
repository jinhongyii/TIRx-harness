"""v2 delta copy of ``tests/numsim/runtime/test_tile_general_semantics.py::test_right_aligned_buffer_broadcast_maps_destination_coordinates``.

Delta: Decision 6, "tile semantics are TVM's dispatch output"
(``docs/development/lowering-inventory.md`` D.2; coordinator ruling quoted in
``docs/development/numsim-behaviour-deltas.md``, "Tile reductions ... Decision 6":
"v2 runs the code TVM's tile dispatch emits for the GPU"). For the f32
``Tx.add`` TVM's dispatch (``elementwise/vec_emit/binary_f32x2.py``, default
``rounding_mode="rz"``) emits ``add.rz.ftz.f32x2``; legacy computed
round-to-nearest. 64 of the 256 sums differ by one ulp (e.g. ``3/7 + 3/5`` is
``1.0285713`` rather than ``1.0285715``). ``sub``/``mul``/``cast`` are unchanged.

Kept: the right-aligned broadcast coordinate mapping for all four ops (the add
expectation uses round-toward-zero).
"""

from __future__ import annotations

import json

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tests.numsim.v2.ports._triage_rz import add_rz_ftz_f32
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


@T.prim_func
def right_aligned_elementwise_broadcast(
    source: T.Buffer((32, 2, 4), "float32"),
    column: T.Buffer((32, 4), "float32"),
    row: T.Buffer((32, 2), "float32"),
    half_column: T.Buffer((32, 4), "float16"),
    output: T.Buffer((32, 4, 2, 4), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_local = T.alloc_buffer((2, 4), "float32", scope="local")
    column_local = T.alloc_buffer((4,), "float32", scope="local")
    row_local = T.alloc_buffer((2, 1), "float32", scope="local")
    half_column_local = T.alloc_buffer((4,), "float16", scope="local")
    add_result = T.alloc_buffer((2, 4), "float32", scope="local")
    sub_result = T.alloc_buffer((2, 4), "float32", scope="local")
    mul_result = T.alloc_buffer((2, 4), "float32", scope="local")
    cast_result = T.alloc_buffer((2, 4), "float32", scope="local")
    for row_index in T.serial(2):
        row_local[row_index, 0] = row[lane, row_index]
        for column_index in T.serial(4):
            source_local[row_index, column_index] = source[lane, row_index, column_index]
    for column_index in T.serial(4):
        column_local[column_index] = column[lane, column_index]
        half_column_local[column_index] = half_column[lane, column_index]

    Tx.add(add_result[:, :], source_local[:, :], column_local[:])
    Tx.sub(sub_result[:, :], source_local[:, :], row_local[:, :])
    Tx.mul(mul_result[:, :], source_local[:, :], row_local[:, :])
    Tx.cast(cast_result[:, :], half_column_local[:])

    for row_index in T.serial(2):
        for column_index in T.serial(4):
            output[lane, 0, row_index, column_index] = add_result[row_index, column_index]
            output[lane, 1, row_index, column_index] = sub_result[row_index, column_index]
            output[lane, 2, row_index, column_index] = mul_result[row_index, column_index]
            output[lane, 3, row_index, column_index] = cast_result[row_index, column_index]


def test_right_aligned_buffer_broadcast_maps_destination_coordinates():
    """Delta copy (Decision 6: dispatched ``add.rz.ftz.f32x2``); see the module docstring."""
    source = np.arange(32 * 2 * 4, dtype=np.float32).reshape(32, 2, 4) / np.float32(7)
    column = np.arange(32 * 4, dtype=np.float32).reshape(32, 4) / np.float32(5)
    row = (np.arange(32 * 2, dtype=np.float32).reshape(32, 2) - 11) / np.float32(3)
    half_column = (column - np.float32(4.25)).astype(np.float16)
    output = np.zeros((32, 4, 2, 4), dtype=np.float32)

    module = v2.transpile(right_aligned_elementwise_broadcast)
    ops = json.loads(module.data)["kernels"][0]["ops"]
    assert {"name": "tirx.ptx.add", "mods": ["rnd=rz", "ftz=ftz", "type=f32x2"]} in ops

    result = v2.Engine().run(
        module,
        {
            "source": source,
            "column": column,
            "row": row,
            "half_column": half_column,
            "output": output,
        },
    )

    expected = np.stack(
        (
            add_rz_ftz_f32(source, np.broadcast_to(column[:, None, :], source.shape)),
            source - row[:, :, None],
            source * row[:, :, None],
            np.broadcast_to(half_column[:, None, :], source.shape),
        ),
        axis=1,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
    # The legacy round-to-nearest sums differ by at most one ulp.
    nearest = source + column[:, None, :]
    assert (result.outputs["output"][:, 0] != nearest).any()
    np.testing.assert_array_max_ulp(result.outputs["output"][:, 0], nearest, maxulp=1)
