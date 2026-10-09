from __future__ import annotations

import re

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def two_direct_output_writers(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = lane
    output[lane] = lane + 100


@T.prim_func
def root_uniform_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        output[lane] = lane
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def root_sync_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = lane
    output[32 + lane] = lane + 100


@T.prim_func
def repeated_split_output_writers(output: T.Buffer((96,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[32 + lane] = lane + 7
    output[64 + lane] = lane + 11


@T.prim_func
def repeated_async_exchange(output: T.Buffer((96,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_shared((96,), "int32")
    if warp == 0:
        shared[32 + lane] = lane + 7
        T.cuda.cta_sync()
        output[32 + lane] = shared[64 + lane] + 5
    else:
        shared[64 + lane] = lane + 11
        T.cuda.cta_sync()
        output[64 + lane] = shared[32 + lane] + 5


@T.prim_func
def inner_split_control_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    uniform_live_in: T.let = (warp // 2) * 2
    if warp == 0:
        live_in: T.let = lane + 7
        output[lane] = live_in + uniform_live_in
        for _outer in T.serial(1):
            for _inner in T.serial(1):
                T.cuda.warp_sync()
                if lane < 16:
                    continue
                output[lane] = live_in + 1
        for _outer in T.serial(1):
            if warp == 0:
                output[lane] = live_in + 1
            else:
                output[lane] = -1
        for _outer in T.serial(1):
            for step in T.serial(2):
                T.cuda.warp_sync()
                if step == 1:
                    break
                output[lane] = output[lane] + 1
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def depth_two_inner_sequence_output_writers(output: T.Buffer((64,), "int32"), lane_limit: T.int32):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        if lane < lane_limit:
            for _outer in T.serial(1):
                for _inner in T.serial(1):
                    for _prefix in T.serial(1):
                        output[lane] = lane
                    T.cuda.warp_sync()
                    output[lane] = output[lane] + 1
        else:
            output[lane] = lane + 50
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def inner_while_sync_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        for _outer in T.serial(1):
            while output[lane] == 0:
                T.cuda.warp_sync()
                output[lane] = lane + 1
    else:
        output[32 + lane] = lane + 100


@T.prim_func
def nested_live_in_output_writers(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    outer_live_in: T.let = lane + 7
    if warp == 0:
        for _outer in T.serial(1):
            while output[lane] == 0:
                T.cuda.warp_sync()
                if warp == 0:
                    if warp < 1:
                        output[lane] = outer_live_in + 1
                    else:
                        output[lane] = -2
                else:
                    output[lane] = -1
    else:
        output[32 + lane] = lane + 100


def _rust_function(source: str, name: str) -> str:
    lines = source.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.removeprefix("pub(super) ").startswith((f"fn {name}(", f"async fn {name}("))
    )
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.match(r"^(?:pub\(super\) )?(?:async )?fn [A-Za-z0-9_]+\(", lines[index])
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_injected_mismatch_reports_buffer_index_and_values(tmp_path):
    module = numsim.transpile(two_direct_output_writers, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})
    expected = result.outputs["output"].copy()
    expected[7] += 1

    report = numsim.compare(result, {"output": expected})

    assert not report.ok
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.output == "output"
    assert mismatch.index == (7,)
    assert mismatch.actual == 107
    assert mismatch.expected == 108

    with pytest.raises(AssertionError) as captured:
        report.require_ok()
    message = str(captured.value)
    assert "first mismatch" in message
    assert "output='output' index=(7,) actual=107 expected=108" in message


@T.prim_func
def varying_while_transfers(output: T.Buffer((32,), "int32")):
    T.device_entry()
    warp = T.warp_id([1])
    lane = T.lane_id([32])
    counter = T.alloc_local((1,), "int32")
    if warp == 0:
        counter[0] = 0
        output[lane] = 0
        while counter[0] < 5:
            counter[0] = counter[0] + 1
            if counter[0] > lane % 3 + 1:
                break
            if counter[0] == 2:
                continue
            output[lane] = output[lane] + counter[0]
        output[lane] = output[lane] + counter[0] * 10


