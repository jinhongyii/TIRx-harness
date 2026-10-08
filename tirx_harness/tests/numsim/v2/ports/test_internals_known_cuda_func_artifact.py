"""v2 copies of two known-CUDA-helper tests from
``tests/analysis_tools/shared/test_known_cuda_func_artifact.py`` (W4-14).

Dropped pins: ``module.rust_source`` substrings (``wrapping_shl(23_u32)``,
``wrapping_add``, ``checked_shl``, ``unwrap_or(0_u32)``). The bit-exact
output assertions are kept; kernels and helper sources copied verbatim.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_COMBINE_SOURCE = r"""
__device__ __forceinline__ float combine_int_frac_ex2(float x_rounded, float frac_ex2) {
  float out;
  asm volatile(
    "{\n\t"
    ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
    "mov.b32 x_rounded_i, %1;\n\t"
    "mov.b32 frac_ex_i, %2;\n\t"
    "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
    "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
    "mov.b32 %0, out_i;\n\t"
    "}\n"
    : "=f"(out) : "f"(x_rounded), "f"(frac_ex2));
  return out;
}
"""


_SHL_U32_CLAMP_SOURCE = r"""
__device__ __forceinline__ unsigned int shl_u32_clamp(unsigned int val, unsigned int shift) {
  unsigned int r;
  asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(val), "r"(shift));
  return r;
}
"""


@T.prim_func
def known_combine_int_frac_ex2(
    rounded: T.Buffer((32,), "float32"),
    fraction: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.func_call(
        "combine_int_frac_ex2",
        rounded[lane],
        fraction[lane],
        source_code=_COMBINE_SOURCE,
        return_type="float32",
    )


@T.prim_func
def known_shl_u32_clamp(
    values: T.Buffer((8,), "uint32"),
    shifts: T.Buffer((8,), "uint32"),
    output: T.Buffer((8,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 8:
        output[lane] = T.cuda.func_call(
            "shl_u32_clamp",
            values[lane],
            shifts[lane],
            source_code=_SHL_U32_CLAMP_SOURCE,
            return_type="uint32",
        )



def test_known_combine_int_frac_ex2_is_bit_exact():
    """Port of ``tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_known_combine_int_frac_ex2_is_bit_exact``."""

    rounded_bits = np.arange(32, dtype=np.uint32) + np.uint32(0x4B40_0000)
    fraction_bits = np.arange(32, dtype=np.uint32) * np.uint32(0x0001_0203) + np.uint32(0x3F80_0000)
    output = np.zeros(32, dtype=np.float32)

    module = v2.transpile(known_combine_int_frac_ex2)
    result = v2.Engine().run(
        module,
        {"rounded": rounded_bits.view(np.float32), "fraction": fraction_bits.view(np.float32), "output": output},
    )

    expected_bits = (rounded_bits << np.uint32(23)) + fraction_bits
    np.testing.assert_array_equal(result.outputs["output"].view(np.uint32), expected_bits)


def test_known_shl_u32_clamp_is_bit_exact():
    """Port of ``tests/analysis_tools/shared/test_known_cuda_func_artifact.py::test_known_shl_u32_clamp_is_bit_exact``."""

    values = np.array([0xFFFF_FFFF, 1, 3, 0x8000_0001, 0xDEAD_BEEF, 7, 9, 11], dtype=np.uint32)
    shifts = np.array([0, 1, 7, 31, 32, 33, 63, 255], dtype=np.uint32)
    output = np.zeros(8, dtype=np.uint32)

    module = v2.transpile(known_shl_u32_clamp)
    result = v2.Engine().run(module, {"values": values, "shifts": shifts, "output": output})

    expected = np.array(
        [
            (int(value) << int(shift)) & 0xFFFF_FFFF if shift < 32 else 0
            for value, shift in zip(values, shifts, strict=True)
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
