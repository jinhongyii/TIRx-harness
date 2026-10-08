from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def scalar_parameter_add(offset: T.int32, output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = offset + lane


@T.prim_func
def second_scalar_parameter_add(offset: T.int32, second: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    second[lane] = offset - lane


@T.prim_func
def scalar_bound_shape(rows: T.int32, input_ptr: T.handle, output_ptr: T.handle):
    input_buffer = T.match_buffer(input_ptr, (rows, 32), "float32")
    output_buffer = T.match_buffer(output_ptr, (rows, 32), "float32")
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output_buffer[0, lane] = input_buffer[0, lane] + T.float32(1)


def test_generated_artifact_rejects_missing_scalar_parameter(tmp_path):
    output = np.zeros(32, dtype=np.int32)
    module = numsim.transpile(scalar_parameter_add, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="missing required bindings.*offset"):
        numsim.Engine().run(module, {"output": output})


