"""v2 port of the legacy TensorMap descriptor-state binding test.

Legacy: tests/numsim/integration/test_bindings.py::test_tensor_map_numpy_is_copyable_descriptor_state
first copied two ``TensorMap(...).numpy()`` images into one plain uint8[256]
array and asserted the legacy ``bindings.prepare_bindings`` payload shape
(``len(descriptor_allocations) == 3``, ``to_payload()["output_allocations"]``).
It then ran a kernel that rewrites the descriptor's swizzle mode with
``tensormap.replace`` for four 128B swizzle variants and asserted that the
independent atomicity field (byte 63) survives, that a re-bind still finds
two descriptor allocations, and that Synccheck and Racecheck are clean.

Dropped: every ``prepare_bindings`` assertion (legacy binder payload shape;
the v2 binder exposes no allocation list). The byte-63 check is kept on the
host array as in legacy and, because v2 ``Engine.run`` does not write
outputs back into host arrays (legacy did, which is what made the host check
observe the kernel), it is also asserted on the bytes the kernel itself
reads back after the replace (``disable_and_dump``, the legacy kernel plus a
copy-out loop). The checker verdicts run through ``v2.synccheck`` /
``v2.racecheck`` on the legacy kernel and inputs.
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import assert_clean, assert_no_incomplete, requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.cases import TensorMap

pytestmark = requires_v2_engine


# --- copied from tests/numsim/integration/test_bindings.py (inline kernel) ---

disable = tvm.script.from_source('''@T.prim_func
def disable(descriptor: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["tensormap_replace.tile.swizzle_mode.global.b1024.b32"](descriptor.ptr_to([0]), 0)
        T.ptx.fence.proxy.tensormap__generic.release.gpu()
''', {"T": T})


# The legacy kernel plus a copy of the rewritten descriptor bytes, so the
# kernel's own view of the descriptor is observable without host write-back.
disable_and_dump = tvm.script.from_source('''@T.prim_func
def disable_and_dump(descriptor: T.Buffer((128,), "uint8"), dump: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["tensormap_replace.tile.swizzle_mode.global.b1024.b32"](descriptor.ptr_to([0]), 0)
        T.ptx.fence.proxy.tensormap__generic.release.gpu()
        for i in T.serial(128):
            dump[i] = descriptor[i]
''', {"T": T})


_SWIZZLES = ("128B", "128B_ATOM_32B", "128B_ATOM_32B_FLIP_8B", "128B_ATOM_64B")


def test_tensor_map_numpy_is_copyable_descriptor_state() -> None:
    """Port of tests/numsim/integration/test_bindings.py::test_tensor_map_numpy_is_copyable_descriptor_state.

    Dropped: the ``prepare_bindings`` descriptor-allocation / payload
    assertions and ``cache_dir`` (see the module docstring).
    """

    base_a = np.arange(12, dtype=np.float32).reshape(3, 4)
    module = v2.transpile(disable)
    dump_module = v2.transpile(disable_and_dump)
    for swizzle in _SWIZZLES:
        descriptor = TensorMap(
            base_a, global_shape=(4, 3), global_strides=(16,), box_shape=(4, 3),
            element_strides=(1, 1), swizzle=swizzle,
        ).numpy()
        tag = descriptor[63]
        inputs = {"descriptor": descriptor}
        v2.Engine().run(module, inputs, outputs=("descriptor",))
        assert descriptor[63] == tag  # The independent atomicity field survives.

        dumped = v2.Engine().run(
            dump_module,
            {"descriptor": descriptor, "dump": np.zeros(128, dtype=np.uint8)},
            outputs=("dump",),
        ).outputs["dump"]
        assert dumped[63] == tag, swizzle

        for checker in (v2.synccheck, v2.racecheck):
            report = checker(disable, inputs)
            assert_clean(report)
            assert_no_incomplete(report)
