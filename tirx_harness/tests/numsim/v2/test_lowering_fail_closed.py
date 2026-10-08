"""Legacy fail-closed rules the lowering keeps (W9 phase 3 `DID NOT RAISE` items)."""

from __future__ import annotations

import pytest

from tirx_harness.numsim.v2.lowering import LoweringUnsupported, lower


def _lower(func):
    from tvm.script import tirx as T

    return lower(func if hasattr(func, "body") else T.prim_func(func), strict=True)


@pytest.mark.parametrize(
    ("module", "name", "reason"),
    [
        ("tests.analysis_tools.shared.test_known_cuda_func_artifact", "modified_combine_int_frac_ex2",
         "does not match the validated"),
        ("tests.analysis_tools.shared.test_known_cuda_func_artifact", "modified_gdn_lg2_approx_ftz",
         "does not match the validated"),
        ("tests.analysis_tools.shared.test_known_cuda_func_artifact", "modified_fma_scale_sub_f32x2",
         "does not match the validated"),
        ("tests.numsim.runtime.test_flashkda_cuda_helpers", "modified_flashkda_rsqrtf", "does not match"),
        ("tests.numsim.runtime.test_flashkda_cuda_helpers", "flashkda_rsqrtf_wrong_dtype", "requires"),
    ],
)
def test_unreviewed_cuda_helper_body_or_signature_fails_closed(module, name, reason):
    import importlib

    with pytest.raises(LoweringUnsupported, match=reason):
        _lower(getattr(importlib.import_module(module), name))


def test_scalar_fp8_identity_reinterpret_fails_closed(lower_source):
    with pytest.raises(LoweringUnsupported, match="raw payload reinterpret is not modeled"):
        lower_source('''
@T.prim_func
def k(source: T.Buffer((32,), "float8_e4m3fn"), output: T.Buffer((32,), "float8_e4m3fn")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    T.warp_id([1])
    output[lane] = T.call_intrin("float8_e4m3fn", "tirx.reinterpret", source[lane])
''')


def test_unknown_tile_config_key_fails_closed():
    from tests.numsim.runtime.test_tile_unary_codegen import tile_unary_unknown_config

    with pytest.raises(LoweringUnsupported, match="unsupported config keys"):
        _lower(tile_unary_unknown_config)


def test_warp_gemm_fragment_layout_off_the_mma_abi_fails_closed():
    from tests.numsim.integration.test_warp_gemm_artifact import _warp_gemm_wrong_a_fragment_layout

    with pytest.raises(LoweringUnsupported, match=r"A fragment layout does not match fixed mma\.sync\.m16n8k16 ABI"):
        _lower(_warp_gemm_wrong_a_fragment_layout)
