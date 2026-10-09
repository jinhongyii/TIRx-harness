"""v2 port of the legacy Synccheck sub-word access test.

Legacy: tests/analysis_tools/racecheck/test_native_sub_word_accesses.py::test_synccheck_accepts_the_ordered_sub_word_kernel
ran the private ``tirx_harness.numsim.checkers._run_synccheck`` on the
barrier-ordered `.b8` store/load kernel and required a clean report. The v2
copy runs ``v2.synccheck`` on the same kernel and inputs. Dropped: the
``cache_dir`` argument (legacy build cache).
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import assert_clean, assert_no_incomplete, requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


# --- copied from tests/analysis_tools/racecheck/test_native_sub_word_accesses.py ---

@T.prim_func
def barrier_ordered_sub_word_store_and_load(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    bytes8 = T.alloc_buffer((64,), "uint8", scope="shared")
    loaded = T.local_scalar("uint8")
    if warp == 0:
        T.ptx.st.shared.b8(bytes8.ptr_to([lane]), T.cast(lane, "uint8"))
    T.cuda.cta_sync()
    if warp == 1:
        T.ptx.ld.shared.b8(loaded, bytes8.ptr_to([lane]))
        output[lane] = T.cast(loaded, "uint32")


def test_synccheck_accepts_the_ordered_sub_word_kernel():
    """Port of tests/analysis_tools/racecheck/test_native_sub_word_accesses.py::test_synccheck_accepts_the_ordered_sub_word_kernel.

    Dropped: ``cache_dir`` only.
    """

    report = v2.synccheck(
        barrier_ordered_sub_word_store_and_load,
        {"output": np.zeros(64, dtype=np.uint32)},
    )
    assert_clean(report)
    assert_no_incomplete(report)
