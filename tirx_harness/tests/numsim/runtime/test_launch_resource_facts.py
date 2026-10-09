"""Launch-model defaults, pointer observations, and CTA-local TMEM lifetimes."""

import numpy as np
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


@T.prim_func
def read_sm_id(output: T.Buffer((3, 32), "uint32")):
    cta = T.cta_id([3])
    lane = T.thread_id([32])
    output[cta, lane] = T.cuda.mov_sreg(32, "smid")


@T.prim_func
def observe_pointer_address(source: T.Buffer((32,), "uint8"), output: T.Buffer((32,), "uint64")):
    lane = T.thread_id([32])
    address = T.reinterpret("uint64", source.ptr_to([lane]))
    output[lane] = address & T.uint64(255)


@T.prim_func
def exclusive_lifecycle(columns: T.uint32, output: T.Buffer((2,), "uint32")):
    T.func_attr({"tirx.cuda_arch": "sm_107a"})
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.thread_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__1.exclusive.sync.aligned.shared__cta.b32(
        T.address_of(address[0]),
        columns,
    )
    if columns == 96:
        # The model deliberately permits ordinary allocation while exclusive is live.
        ordinary = T.alloc_buffer((1,), "uint32", scope="shared")
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(ordinary[0]),
            32,
        )
        T.ptx.tcgen05.dealloc.cta_group__1.exclusive.sync.aligned.b32(ordinary[0], 32)
    value = T.alloc_local((1,), "uint32")
    value[0] = T.Cast("uint32", cta + 1)
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    if lane == 0:
        output[cta] = value[0]
    T.ptx.tcgen05.dealloc.cta_group__1.exclusive.sync.aligned.b32(address[0], columns)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


def test_smid_defaults_to_zero_in_execution_and_both_checkers(tmp_path):
    inputs = {"output": np.full((3, 32), 99, dtype=np.uint32)}
    for checker in (synccheck, racecheck):
        checker(read_sm_id, inputs).require_clean()
    module = numsim.transpile(read_sm_id, cache_dir=tmp_path)
    result = numsim.Engine().run(module, inputs)
    np.testing.assert_array_equal(result.outputs["output"], np.zeros((3, 32), dtype=np.uint32))


