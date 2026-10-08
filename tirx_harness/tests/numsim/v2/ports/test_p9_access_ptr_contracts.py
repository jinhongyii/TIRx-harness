"""v2 copies of the ``tvm_access_ptr`` contract tests of
``tests/numsim/integration/test_direct_memory_artifact.py`` (W12).

A ``tvm_access_ptr(type, data, offset, extent, rw_mask)`` states which
accesses the IR makes through the pointer; TVM's analyses rely on it, the
GPU does not check it. Legacy failed closed on every violation at run time;
v2 checks the same contract statically, so the rejection is an
``UnsupportedTIRxError`` at transpile (stage moved, legacy message fragments
kept) that names the call site.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


@T.prim_func
def store_through_read_only_access_ptr(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.st.global_.u32(output.access_ptr("r", offset=lane, extent=1), T.uint32(7))


@T.prim_func
def load_through_write_only_access_ptr(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.global_.u32(output[lane], source.access_ptr("w", offset=lane, extent=1))


@T.prim_func
def nested_access_ptr_cannot_widen_permissions(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    read_only = output.access_ptr("r", offset=lane, extent=1)
    widened = T.tvm_access_ptr(T.type_annotation("uint32"), read_only, 0, 1, 2)
    T.ptx.st.global_.u32(widened, T.uint32(9))


@T.prim_func
def write_through_read_only_decl_buffer(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias_data: T.let[
        T.Var(name="read_only_alias", ty=PointerType(PrimType("uint32"), "global"))
    ] = source.access_ptr("r", ptr_type="uint32", offset=0, extent=32)
    alias = T.decl_buffer((32,), "uint32", data=alias_data, scope="global")
    alias[lane] = output[lane]


@T.prim_func
def permitted_decl_buffer_accesses(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    read_data: T.let[
        T.Var(name="permitted_read_alias", ty=PointerType(PrimType("uint32"), "global"))
    ] = source.access_ptr("r", ptr_type="uint32", offset=0, extent=32)
    write_data: T.let[
        T.Var(name="permitted_write_alias", ty=PointerType(PrimType("uint32"), "global"))
    ] = output.access_ptr("w", ptr_type="uint32", offset=0, extent=32)
    read_alias = T.decl_buffer((32,), "uint32", data=read_data, scope="global")
    write_alias = T.decl_buffer((32,), "uint32", data=write_data, scope="global")
    write_alias[lane] = read_alias[lane] + T.uint32(5)


@T.prim_func
def oversized_decl_buffer_from_access_ptr(source: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    alias_data: T.let[T.Var(name="short_alias", ty=PointerType(PrimType("uint32"), "global"))] = (
        source.access_ptr("r", ptr_type="uint32", offset=0, extent=1)
    )
    _alias = T.decl_buffer((2,), "uint32", data=alias_data, scope="global")


@T.prim_func
def load_after_access_ptr_extent(
    source: T.Buffer((64,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    one_element = source.access_ptr("r", offset=lane, extent=1)
    one_past = T.ptr_byte_offset(one_element, T.uint32(4), "uint32")
    T.ptx.ld.global_.u32(output[lane], one_past)



@T.prim_func
def dps_read_only_destination(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = output.access_ptr("r", offset=lane, extent=1)
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@T.prim_func
def dps_destination_after_access_extent(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    bounded = output.access_ptr("w", offset=lane, extent=1)
    destination = T.ptr_byte_offset(bounded, T.uint32(4), "float32")
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (store_through_read_only_access_ptr, "store through non-writable physical pointer"),
        (load_through_write_only_access_ptr, "load through non-readable physical pointer"),
        (nested_access_ptr_cannot_widen_permissions, "cannot add write access"),
        (write_through_read_only_decl_buffer, "write through non-writable DeclBuffer view"),
        (oversized_decl_buffer_from_access_ptr, "outside tvm_access_ptr range"),
    ],
    ids=["store_through_read_only_access_ptr", "load_through_write_only_access_ptr",
         "nested_access_ptr_cannot_widen_permissions", "write_through_read_only_decl_buffer",
         "oversized_decl_buffer_from_access_ptr"],
)
def test_access_ptr_contract_fails_closed(kernel, message):
    """Copy of ``test_direct_memory_artifact.py::test_access_ptr_contract_fails_closed`` (all params)."""

    with pytest.raises(UnsupportedTIRxError, match=message):
        v2.transpile(kernel)


def test_decl_buffer_views_preserve_permitted_accesses():
    """Positive control: reads through ``"r"`` and writes through ``"w"`` views run."""

    source = np.arange(32, dtype=np.uint32)
    result = v2.Engine().run(v2.transpile(permitted_decl_buffer_accesses),
                             {"source": source, "output": np.zeros(32, dtype=np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], source + np.uint32(5))


def test_dps_preserves_pointer_access_contracts():
    """Copy of ``test_artifact_build.py::test_dps_preserves_pointer_access_contracts``
    (stage moved to transpile, as above) and its control
    ``test_dps_integer_pointer_arithmetic_does_not_carry_access_ptr_extent``."""

    with pytest.raises(UnsupportedTIRxError, match="non-writable physical pointer"):
        v2.transpile(dps_read_only_destination)
    result = v2.Engine().run(v2.transpile(dps_destination_after_access_extent),
                             {"output": np.zeros(33, dtype=np.float32)})
    assert result.status["kind"] == "completed"
