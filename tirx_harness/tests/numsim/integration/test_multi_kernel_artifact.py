from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.support.kernels import raw_tma_roundtrip
from tests.numsim.support.multi_kernels import (
    ambiguous_alias_first,
    ambiguous_alias_second,
    consume_intermediate,
    same_names_first,
    same_names_second,
    typed_pointer_first,
    typed_pointer_second,
    write_intermediate,
)
from tirx_harness import numsim


@T.prim_func
def export_global_pointer_words(
    source: T.Buffer((32,), "uint32"), pointers: T.Buffer((32,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointers[lane] = T.reinterpret("uint64", source.ptr_to([lane]))


@T.prim_func
def consume_global_pointer_words(
    pointers: T.Buffer((32,), "uint64"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", pointers[lane]))


def _same_name_inputs(*, shared_intermediate: bool):
    input0 = np.arange(32, dtype=np.float32)
    input1 = np.arange(32, dtype=np.float32) + np.float32(100)
    intermediate0 = np.zeros(32, dtype=np.float32)
    intermediate1 = intermediate0 if shared_intermediate else np.full(32, 7, dtype=np.float32)
    output0 = np.zeros(32, dtype=np.float32)
    output1 = np.zeros(32, dtype=np.float32)
    return {
        "k0:input": input0,
        "k0:intermediate": intermediate0,
        "k0:output": output0,
        "k0:scale": 3,
        "k1:input": input1,
        "k1:intermediate": intermediate1,
        "k1:output": output1,
        "k1:scale": np.float32(0.5),
    }


def test_same_local_names_use_distinct_phase_bindings(tmp_path):
    module = numsim.transpile([same_names_first, same_names_second], cache_dir=tmp_path)
    inputs = _same_name_inputs(shared_intermediate=False)

    result = numsim.Engine().run(module, inputs)

    input0 = inputs["k0:input"]
    input1 = inputs["k1:input"]
    np.testing.assert_array_equal(result.outputs["k0:intermediate"], input0 * 3)
    np.testing.assert_array_equal(result.outputs["k0:output"], input0 + 1)
    np.testing.assert_array_equal(result.outputs["k1:output"], input1 + np.float32(3.5))
    assert set(result.outputs) == {
        "k0:input",
        "k0:intermediate",
        "k0:output",
        "k1:input",
        "k1:intermediate",
        "k1:output",
    }


def test_cross_phase_sharing_requires_the_same_explicit_backing(tmp_path):
    module = numsim.transpile([same_names_first, same_names_second], cache_dir=tmp_path)
    inputs = _same_name_inputs(shared_intermediate=True)

    result = numsim.Engine().run(module, inputs)

    input0 = inputs["k0:input"]
    input1 = inputs["k1:input"]
    np.testing.assert_array_equal(result.outputs["k0:intermediate"], input0 * 3)
    np.testing.assert_array_equal(result.outputs["k1:output"], input1 + input0 * np.float32(1.5))


def test_single_kernel_keeps_unqualified_binding_names(tmp_path):
    module = numsim.transpile(same_names_first, cache_dir=tmp_path)
    source = np.arange(32, dtype=np.float32)
    intermediate = np.zeros(32, dtype=np.float32)
    output = np.zeros(32, dtype=np.float32)

    result = numsim.Engine().run(
        module,
        {
            "input": source,
            "intermediate": intermediate,
            "output": output,
            "scale": 2,
        },
    )

    assert set(result.outputs) == {"input", "intermediate", "output"}
    np.testing.assert_array_equal(result.outputs["intermediate"], source * 2)
    np.testing.assert_array_equal(result.outputs["output"], source + 1)


