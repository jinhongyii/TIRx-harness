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


def test_tvm_pair_cast_helper_lowers_to_round_to_nearest_casts():
    """W4-14: `tvm_builtin_cast_float32x2_float16x2(dst, src)` -> 2 loads, 2 Rn casts, 2 stores."""
    from tests.numsim.runtime.test_tile_general_semantics import right_aligned_elementwise_broadcast

    program = _lower(right_aligned_elementwise_broadcast)
    casts = [i for i in program.code if i.variant == "Cast" and i.fields["from"].elem == "F16"]
    assert casts and all(i.rnd == "Rn" for i in casts)
    assert not any(op.name.startswith("tirx.cuda.func_call.tvm_builtin_cast_") for op in program.ops)


def test_smem_desc_make_lo_uniform_is_a_lane_zero_shuffle():
    """W4-14: the reviewed broadcast lowers to Shfl(Idx, lane 0) of the low word."""
    from tests.numsim.integration.test_opaque_helper_artifact import smem_descriptor_make_lo_uniform_helper

    program = _lower(smem_descriptor_make_lo_uniform_helper)
    shfl = [i for i in program.code if i.variant == "Shfl"]
    assert len(shfl) == 1 and shfl[0].mode == "Idx" and shfl[0].ty.elem == "U32"


def test_spdecompress_rejects_overlapping_or_unprovable_registers():
    from tests.numsim.v2.ports.test_p6c_ptx_spdecompress_registry import (
        spdecompress_aliased_register_views,
        spdecompress_dynamic_register_index,
    )

    with pytest.raises(LoweringUnsupported, match="overlap"):
        _lower(spdecompress_aliased_register_views)
    # A dynamic index is not proof of overlap: the unproven pair becomes a
    # run-time Assert on the concrete byte ranges (fails only when they overlap).
    program = _lower(spdecompress_dynamic_register_index)
    asserts = [i for i in program.code if i.variant == "Assert"]
    messages = [program.strings[i.msg] for i in asserts]
    assert sum("undefined register overlap" in m for m in messages) == 2, messages


def test_spdecompress_register_disjointness_from_bounds_and_memory_operands():
    """Global operands are not physical registers (legacy
    ``test_ptx_spdecompress_matches_low_bit_first_sparse_scatter``); local
    operands whose index bounds keep them apart need no check."""
    from tests.numsim.runtime.test_ptx_spdecompress import ptx_spdecompress_b8_b4_2_4_x2
    from tvm.script import tirx as T

    @T.prim_func
    def bounded(index: T.int32):
        T.device_entry()
        storage = T.alloc_buffer((8,), "uint32", scope="local")
        for i in range(2):
            T.ptx["spdecompress.b8.b4.sp::1:2.x2"](storage[i], storage[4 + T.min(T.max(index, 0), 3)], storage[2])

    for kernel in (ptx_spdecompress_b8_b4_2_4_x2, bounded):
        program = _lower(kernel)
        messages = [program.strings[i.msg] for i in program.code if i.variant == "Assert" and i.msg is not None]
        assert not any("register overlap" in m for m in messages), messages
