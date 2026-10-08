"""Data-path lowering rules settled from the v2 xfail inventory (W12):
``s_tir.ldg32``, the ``cuda.ldg`` overload set, vector bitwise nodes (delta
D11), and the architecture requirements of shared-memory descriptors.
Assertions are on Program contents, never on text."""

from __future__ import annotations

import pytest

from tirx_harness.numsim.v2.lowering import program_builder as pb
from tirx_harness.numsim.v2.lowering.ir_walk import LoweringUnsupported

from ._program import all_of, const, only, pc_of


def test_ldg32_loads_guarded_lanes_through_the_readonly_path_and_zeroes_the_rest(lower_source):
    program = lower_source('''
@T.prim_func
def k(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "float32")):
    T.attr({"tirx.device_entry": T.bool(True)})
    T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_buffer((32,), "float32", scope="local")
    T.evaluate(T.s_tir.ldg32(local.data, lane < 16, source[lane], lane))
    output[lane] = local[lane]
''')
    branch = only(program, "If")
    load = only(program, "LoadAddr")
    assert (load.ty, load.space, load.mods["nc"]) == (pb.Ty("F32"), "Global", True)
    otherwise = only(program, "Else")
    local = [b.name for b in program.buffers].index("local")
    stores = [i for i in all_of(program, "Store") if i.buf == local]
    assert len(stores) == 2
    # then: the loaded value; else: +0.0 (TVM codegen `@!p mov.b32 r, 0`).
    assert pc_of(program, branch) < pc_of(program, load) < pc_of(program, stores[0]) < pc_of(program, otherwise)
    assert pc_of(program, otherwise) < pc_of(program, stores[1]) and const(program, stores[1].value) == 0


@pytest.mark.parametrize("dtype", ["bool", "float8_e4m3fn", "float8_e4m3fnx4", "float16x4", "bfloat16x4"])
def test_cuda_ldg_outside_the_ldg_overload_set_fails_closed(lower_source, dtype):
    with pytest.raises(LoweringUnsupported, match="__ldg overload"):
        lower_source(f'''
@T.prim_func
def k(source: T.Buffer((32,), "{dtype}"), output: T.Buffer((32,), "{dtype}")):
    T.attr({{"tirx.device_entry": T.bool(True)}})
    T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.ldg(source.ptr_to([lane]), "{dtype}")
''')


def test_cuda_ldg_pointer_must_point_at_the_loaded_dtype(lower_source):
    with pytest.raises(LoweringUnsupported, match="does not match the loaded dtype"):
        lower_source('''
@T.prim_func
def k(source: T.Buffer((64,), "float32"), output: T.Buffer((32,), "float32x2")):
    T.attr({"tirx.device_entry": T.bool(True)})
    T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.ldg(source.ptr_to([lane * 2]), "float32x2")
''')


@pytest.mark.parametrize("operation,op", [("bitwise_and", "And"), ("bitwise_or", "Or"),
                                          ("bitwise_xor", "Xor"), ("shift_left", "Shl"),
                                          ("shift_right", "Shr")])
def test_vector_bitwise_node_is_one_lane_wise_binary(lower_source, operation, op):
    program = lower_source(f'''
@T.prim_func
def k(a: T.Buffer((32,), "int32x2"), out: T.Buffer((32,), "int32x2")):
    T.attr({{"tirx.device_entry": T.bool(True)}})
    lane = T.lane_id([32])
    x: T.let = a[lane]
    out[lane] = T.{operation}(x, x)
''')
    binary = [i for i in all_of(program, "Binary") if i.op == op]
    assert len(binary) == 1 and binary[0].ty == pb.Ty("S32", 2)


def test_vector_bitwise_not_fails_closed(lower_source):
    with pytest.raises(LoweringUnsupported, match="bitwise_not on vector"):
        lower_source('''
@T.prim_func
def k(a: T.Buffer((32,), "int32x2"), out: T.Buffer((32,), "int32x2")):
    T.attr({"tirx.device_entry": T.bool(True)})
    lane = T.lane_id([32])
    out[lane] = T.bitwise_not(a[lane])
''')
