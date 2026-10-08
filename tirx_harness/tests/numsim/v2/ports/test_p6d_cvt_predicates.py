"""v2 delta copy of ``tests/numsim/runtime/test_cvt_predicates.py::test_stochastic_cvt_predicates_preserve_random_bits``.

The kernel issues ``cvt.rs[.relu].satfinite.{e2m1x4,e4m3x4,e5m2x4}.f32`` in three
predicate modes per form: (0) ``pred=lane%2==0, preserve_dst=True``, (1)
``pred=lane%2==0`` (no ``preserve_dst``), (2) ``pred=False, preserve_dst=True``.

Delta, mode 1 inactive (odd) lanes only: TVM renders a predicated op without
``preserve_dst`` as ``*_pred_undef`` (``tvm/backend/cuda/ptx/render.py``: the
destination is bound write-only, "the caller will not consume the inactive
value"), so the value is undefined on hardware. Legacy wrote 0 there; v2 keeps
the destination for every guarded op (``CONTRACT_REQUESTS.md`` "W1
(2026-10-08): W2-20 lowering items", item 1: "A ``@p`` op now emits
``keep_dst: true``"), so those lanes read the 0x5A5A sentinel the kernel
stored before the op. All 768 differing words are these lanes.

Kept unchanged: the GPU goldens for active lanes in modes 0 and 1, the
preserved sentinel in mode 0 / mode 2, and the clean racecheck/synccheck
verdicts (legacy ``run_checked``).
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.microtests.cases.ptx_cvt_narrow_goldens import F32_SOURCE, GOLDENS, RBITS_SOURCE
from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_SENTINEL = 0x5A5A


def stochastic_predicate_case():
    """Copy of the legacy builder; the mode-1 inactive lanes expect the kept sentinel."""

    lines = []
    expected = np.full((18, 256), _SENTINEL, np.uint32)
    forms = [
        (destination, relu) for destination in ("e2m1x4", "e4m3x4", "e5m2x4") for relu in (False, True)
    ]
    for index, (destination, relu) in enumerate(forms):
        spelling = f"cvt.rs{'.relu' if relu else ''}.satfinite.{destination}.f32"
        golden = GOLDENS[f"{destination}.f32.rs{'.relu' if relu else ''}"]
        dtype = "uint16" if destination == "e2m1x4" else "uint32"
        lines.append(f'        dst_{index} = T.alloc_local((1,), "{dtype}")')
        for mode in range(3):
            row = 3 * index + mode
            offset = "256" if mode == 2 else "source_index"
            arguments = ", ".join(
                f"source[T.Select({offset} < 256, ({offset} + {part}) % 256, 256)]" for part in range(4)
            )
            predicate = "False" if mode == 2 else "lane % 2 == 0"
            preserve = ", preserve_dst=True" if mode != 1 else ""
            lines += [
                f"        dst_{index}[0] = T.{dtype}(0x5A5A)",
                f'        T.ptx["{spelling}"](dst_{index}[0], {arguments}, random[{offset}], '
                f"pred={predicate}{preserve})",
                f'        output[{row}, i] = T.cast(dst_{index}[0], "uint32")',
            ]
            if mode != 2:
                expected[row, ::2] = golden[::2]
            # mode 1, odd lanes: legacy expected 0; v2 keeps the sentinel (see module docstring).
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(source: T.Buffer((256,), "float32"), random: T.Buffer((256,), "float32"), output: T.Buffer((18, 256), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        i = step * 32 + lane
        source_index = T.Select(lane % 2 == 0, i, 256)
{chr(10).join(lines)}
""",
        {"T": T},
    )
    inputs = dict(
        source=F32_SOURCE.view(np.float32),
        random=RBITS_SOURCE.view(np.float32),
        output=np.zeros_like(expected),
    )
    return kernel, inputs, expected


def test_stochastic_cvt_predicates_preserve_random_bits():
    """Delta copy (W1 W2-20 item 1, guarded PTX keeps its destination); see the module docstring."""

    kernel, inputs, expected = stochastic_predicate_case()
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(kernel, inputs)
        assert report.verdict == "clean", report.format()
    result = v2.Engine().run(v2.transpile(kernel), inputs)
    np.testing.assert_array_equal(result.outputs["output"], expected)
