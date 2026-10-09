"""Engine instructions emitted for public packed PTX ``cvt`` calls."""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tirx_harness import numsim


def _cvt_call(spelling: str, destination_dtype: str, *arguments: str):
    func = tvm.script.from_source(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    destination = T.local_scalar("{destination_dtype}")
    T.ptx["{spelling}"](destination, {", ".join(arguments)})
""",
        {"T": T},
    )
    calls = []

    def visit(node: object) -> None:
        name = str(getattr(getattr(node, "op", None), "name", ""))
        if type(node).__name__ == "Call" and (
            name == "tirx.ptx.cvt" or name.startswith("tirx.ptx.cvt_")
        ):
            calls.append(node)

    structural_walk(func.body, ((Expr, Stmt), visit))
    assert len(calls) == 1
    return func, calls[0]


def _packed_head(source: str, destination: str, *modes: str) -> str:
    """The ``v2::reg::cvt`` head of a packed conversion with ``modes``."""

    variant = "v2::reg::variant::"
    spelled = ", ".join(variant + mode for mode in modes)
    return (
        f"v2::reg::cvt::<{variant}Cvt<{variant}{source}, {variant}{destination}, "
        f"{variant}PackedMode<{spelled}>>>"
    )


def test_f6_form_transpiles_and_preserves_exact_half_bits(tmp_path):
    @T.prim_func
    def f6_cvt(source: T.Buffer((32,), "uint16"), output: T.Buffer((32,), "uint32")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        T.ptx["cvt.rn.f16x2.e2m3x2"](output[lane], source[lane])

    # E2M3 encodes 1, 2 and 4 as 0x08, 0x10 and 0x18; sign is bit5.
    source = np.tile(np.array([0, 0x0810, 0x2018, 0x2808], np.uint16), 8)
    expected = np.tile(np.array([0, 0x3C004000, 0x80004400, 0xBC003C00], np.uint32), 8)
    result = numsim.Engine().run(
        numsim.transpile(f6_cvt, cache_dir=tmp_path),
        {"source": source, "output": np.zeros(32, np.uint32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    ("spelling", "destination_dtype", "arguments"),
    [
        pytest.param(
            "cvt.rn.satfinite.ftz.e4m3x2.f16x2",
            "uint16",
            ("T.uint32(0x3C003C00)",),
            id="ftz_e4m3x2_from_f16x2",
        ),
        pytest.param(
            "cvt.rn.satfinite.ftz.e2m1x2.f32",
            "uint8",
            ("T.float32(1.0)", "T.float32(2.0)"),
            id="ftz_e2m1x2_from_f32",
        ),
        pytest.param(
            "cvt.rz.f32.f32",
            "float32",
            ("T.float32(1.5)",),
            id="directed_rounding_scalar",
        ),
    ],
)
def test_non_table_cvt_neighbours_are_rejected_by_target_parser(
    spelling, destination_dtype, arguments
):
    with pytest.raises(tvm.error.DiagnosticError):
        _cvt_call(spelling, destination_dtype, *arguments)
