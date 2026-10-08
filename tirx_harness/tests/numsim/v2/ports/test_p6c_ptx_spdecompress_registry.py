"""v2 ports of the register-overlap rejections in
``tests/numsim/registry/test_ptx_spdecompress_registry.py``.

Legacy rejected these at Rust emission (``emit_rust_module(analyze(...))``)
with ``UnsupportedTIRxError``. The ports keep the fail-closed contract
through the public surface: the overlapping ``spdecompress`` must be
rejected by ``v2.transpile`` or stop the run with ``v2.ExecutionError``.
Message text is not pinned. Kernels are copied verbatim.
"""

from __future__ import annotations

import pytest
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine, v2_gap
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


@T.prim_func
def spdecompress_aliased_register_views():
    T.device_entry()
    storage = T.alloc_buffer((3,), "uint32", scope="local")
    alias = T.decl_buffer((2,), "uint32", data=storage.data, elem_offset=1, scope="local")
    T.ptx["spdecompress.b8.b4.sp::1:2.x2"](storage[1], alias[0], storage[2])


@T.prim_func
def spdecompress_dynamic_register_index(index: T.int32):
    T.device_entry()
    storage = T.alloc_buffer((3,), "uint32", scope="local")
    T.ptx["spdecompress.b8.b4.sp::1:2.x2"](storage[index], storage[1], storage[2])


@v2_gap(
    "v2 transpiles and completes spdecompress whose data register storage[1] "
    "and mdata register alias[0] are the same physical register; legacy "
    "rejected it ('undefined register overlap ... aliased')"
)
def test_spdecompress_rejects_same_physical_register_through_alias_views():
    """Port of ``tests/numsim/registry/test_ptx_spdecompress_registry.py::test_spdecompress_rejects_same_physical_register_through_alias_views``."""

    with pytest.raises((UnsupportedTIRxError, v2.ExecutionError)):
        v2.Engine().run(v2.transpile(spdecompress_aliased_register_views), {})


@v2_gap(
    "v2 transpiles spdecompress with a dynamic register index and completes "
    "the run even when index=1 makes data and mdata the same register; legacy "
    "rejected the unprovable overlap at transpile ('cannot prove disjoint "
    "physical register')"
)
def test_spdecompress_rejects_unknown_dynamic_register_overlap():
    """Port of ``tests/numsim/registry/test_ptx_spdecompress_registry.py::test_spdecompress_rejects_unknown_dynamic_register_overlap``.

    Legacy rejected statically; v2 may instead reject the overlapping
    concrete invocation (``index=1``) at run time.
    """

    with pytest.raises((UnsupportedTIRxError, v2.ExecutionError)):
        v2.Engine().run(v2.transpile(spdecompress_dynamic_register_index), {"index": 1})
