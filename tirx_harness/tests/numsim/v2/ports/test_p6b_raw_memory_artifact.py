"""v2 ports of ``tests/numsim/integration/test_raw_memory_artifact.py`` tests
that pinned ``module.rust_source``, legacy error text or used the legacy
``analyze`` / ``emit_rust_module`` front end. Kernels are copied verbatim (from
the legacy file or ``tests/numsim/support/kernels.py``).
``analyze(kernel).unsupported == ()`` becomes "``v2.transpile`` accepts the
kernel"; generated-Rust text pins are dropped.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def _first_stop(error: v2.ExecutionError) -> dict:
    """The diagnostic ``Engine.run`` raised for (same selection as run.py)."""

    stops = [
        d
        for d in error.diagnostics
        if d.get("status") in ("error", "incomplete") and d.get("reason") != "subset_execution"
    ]
    assert stops, error.diagnostics
    return stops[0]


@T.prim_func
def raw_shared_v2_ordered_destination_loads(
    source: T.Buffer((64,), "uint32"),
    relaxed: T.Buffer((64,), "uint32"),
    acquired: T.Buffer((64,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    relaxed_pair = T.alloc_local((2,), "uint32")
    acquired_pair = T.alloc_local((2,), "uint32")
    for index in T.unroll(2):
        shared[lane * 2 + index] = source[lane * 2 + index]
    T.cuda.warp_sync()
    T.ptx.ld.relaxed.cta.shared.v2.u32(relaxed_pair[0], relaxed_pair[1], shared.ptr_to([lane * 2]))
    T.ptx.ld.acquire.cta.shared.v2.u32(
        acquired_pair[0], acquired_pair[1], shared.ptr_to([lane * 2])
    )
    for index in T.unroll(2):
        relaxed[lane * 2 + index] = relaxed_pair[index]
        acquired[lane * 2 + index] = acquired_pair[index]


@T.prim_func
def raw_shared_v2_volatile_destination_load(
    source: T.Buffer((64,), "uint32"), output: T.Buffer((64,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    pair = T.alloc_local((2,), "uint32")
    for index in T.unroll(2):
        shared[lane * 2 + index] = source[lane * 2 + index]
    T.cuda.warp_sync()
    T.ptx.ld.volatile.shared.v2.u32(pair[0], pair[1], shared.ptr_to([lane * 2]))
    for index in T.unroll(2):
        output[lane * 2 + index] = pair[index]


@T.prim_func
def raw_shared_misaligned_ordered_b128_load(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "uint32", scope="shared", align=16)
    loaded = T.alloc_local((1,), "uint128")
    if lane == 0:
        for index in T.serial(8):
            shared[index] = T.uint32(index + 1)
        T.ptx.ld.acquire.cta.shared.b128(
            loaded[0],
            T.cuda.cvta_generic_to_shared(shared.ptr_to([1])),
        )
        for index in T.serial(4):
            output[index] = loaded.view("uint32")[index]


@T.prim_func
def raw_volatile_b128_uninitialized(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    loaded = T.alloc_local((1,), "uint128")
    if lane == 0:
        T.ptx.ld.volatile.shared.b128(loaded[0], shared.ptr_to([0]))
        for index in T.serial(4):
            output[index] = loaded.view("uint32")[index]


@T.prim_func
def pointer_derived_shared_raw_roundtrip(
    source: T.Buffer((32,), "uint32"),
    loaded: T.Buffer((32,), "uint32"),
    aliased: T.Buffer((32,), "uint32"),
    addresses: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((192,), "uint8", scope="shared")
    base_data: T.let[T.Var(name="raw_shared_base", ty=PointerType(PrimType("void"), "shared"))] = (
        T.reinterpret(PointerType(PrimType("void"), "shared"), shared.ptr_to([32]))
    )
    base = T.decl_buffer((36,), "uint32", data=base_data, scope="shared")
    stage_data: T.let[
        T.Var(name="raw_shared_stage", ty=PointerType(PrimType("uint32"), "shared"))
    ] = T.ptr_byte_offset(base.data, T.uint32(16), "uint32")
    stage = T.decl_buffer((32,), "uint32", data=stage_data, scope="shared")
    T.ptx.st.shared.u32(stage.ptr_to([lane]), source[lane])
    T.cuda.warp_sync()
    T.ptx.ld.shared.u32(loaded[lane], stage.ptr_to([lane]))
    aliased[lane] = base[4 + lane]
    addresses[lane] = T.cuda.cvta_generic_to_shared(stage.ptr_to([lane]))


@T.prim_func
def get_tmem_addr_lane_values(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lane_u32 = T.cast(lane, "uint32")
    output[lane] = T.cuda.get_tmem_addr(T.uint32(0xFFF0FFF0), T.int32(32), lane_u32 * T.uint32(3))
    output[32 + lane] = T.cuda.get_tmem_addr(T.int32(0x00100010), T.int32(-32), T.int32(0) - lane)


@T.prim_func
def raw_load_rejects_integer_address(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    forged: T.let[T.Var(name="forged_pointer", ty=PointerType(PrimType("uint32")))] = T.reinterpret(
        PointerType(PrimType("uint32")), T.uint64(64)
    )
    T.ptx.ld.global_.u32(output[lane], forged)


def test_get_tmem_addr_packs_wrapped_row_and_column_offsets_per_lane():
    """Port of ``tests/numsim/integration/test_raw_memory_artifact.py::test_get_tmem_addr_packs_wrapped_row_and_column_offsets_per_lane``.

    Dropped pins: ``"get_tmem_addr("`` in and ``"fn get_tmem_addr("`` not in
    ``module.rust_source``.
    """

    output = np.zeros(64, dtype=np.uint32)

    module = v2.transpile(get_tmem_addr_lane_values)
    result = v2.Engine().run(module, {"output": output})

    lanes = np.arange(32, dtype=np.uint32)
    expected = np.empty(64, dtype=np.uint32)
    expected[:32] = (np.uint32(0x0010) << np.uint32(16)) | (
        (np.uint32(0xFFF0) + lanes * np.uint32(3)) & np.uint32(0xFFFF)
    )
    expected[32:] = (np.uint32(0xFFF0) << np.uint32(16)) | (
        (np.uint32(0x0010) - lanes) & np.uint32(0xFFFF)
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ordered_b128_load_retains_16_byte_alignment_contract():
    """Port of ``tests/numsim/integration/test_raw_memory_artifact.py::test_ordered_b128_load_retains_16_byte_alignment_contract``.

    Dropped: ``match="b128 requires 16-byte alignment"`` (legacy text).
    Asserted: ``v2.ExecutionError`` whose stopping diagnostic is an
    ``error`` of kind ``misaligned``.
    """

    module = v2.transpile(raw_shared_misaligned_ordered_b128_load)

    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(4, dtype=np.uint32)})
    stop = _first_stop(caught.value)
    assert stop["status"] == "error", stop
    assert stop["kind"] == "misaligned", stop


def test_ordered_v2_destination_loads_keep_their_memory_semantics():
    """Port of ``tests/numsim/integration/test_raw_memory_artifact.py::test_ordered_v2_destination_loads_keep_their_memory_semantics``.

    `.relaxed`, `.acquire` and `.volatile` all take `.vec` in PTX.

    Dropped pins: the ``v2::mem::variant::{Relaxed,Acquire}<..Cta>`` and
    ``v2::mem::variant::Volatile`` strings in ``module.rust_source``
    (legacy generated code). The loaded values are asserted.
    """

    source = (np.arange(64, dtype=np.uint32) * np.uint32(0x9E3779B1)) ^ np.uint32(0x1234_5678)
    module = v2.transpile(raw_shared_v2_ordered_destination_loads)
    result = v2.Engine().run(
        module,
        {
            "source": source,
            "relaxed": np.zeros(64, dtype=np.uint32),
            "acquired": np.zeros(64, dtype=np.uint32),
        },
    )

    for name in ("relaxed", "acquired"):
        np.testing.assert_array_equal(result.outputs[name], source)

    volatile_module = v2.transpile(raw_shared_v2_volatile_destination_load)
    volatile_result = v2.Engine().run(
        volatile_module, {"source": source, "output": np.zeros(64, dtype=np.uint32)}
    )
    np.testing.assert_array_equal(volatile_result.outputs["output"], source)


def test_volatile_b128_load_preserves_uninitialized_review():
    """Port of ``tests/numsim/integration/test_raw_memory_artifact.py::test_volatile_b128_load_preserves_uninitialized_review``.

    Dropped pin: ``"v2::mem::variant::Volatile"`` in ``module.rust_source``.
    Kept: both checkers report ``review`` with only ``uninitialized_read``
    findings, and the NumSim run reads zeros with verdict ``review`` and
    only ``uninitialized_read`` diagnostics.
    """

    for checker in (v2.synccheck, v2.racecheck):
        report = checker(raw_volatile_b128_uninitialized, {"output": np.zeros(4, np.uint32)})
        assert report.verdict == "review", report.format()
        assert {finding.kind for finding in report.findings} == {"uninitialized_read"}
    module = v2.transpile(raw_volatile_b128_uninitialized)
    result = v2.Engine().run(module, {"output": np.full(4, 0xDEADBEEF, np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], np.zeros(4, np.uint32))
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}


def test_pointer_derived_shared_views_raw_memory_and_cvta_preserve_aliasing():
    """Port of ``tests/numsim/integration/test_raw_memory_artifact.py::test_pointer_derived_shared_views_raw_memory_and_cvta_preserve_aliasing``.

    Dropped: legacy ``analyze(...).unsupported == ()`` (replaced by
    ``v2.transpile`` accepting the kernel) and the ``expect_harness_surface``
    fixture (it repeated the ``addresses`` assertion below).
    """

    source = (np.arange(32, dtype=np.uint32) * np.uint32(17)) ^ np.uint32(0xA5A55A5A)
    loaded = np.zeros(32, dtype=np.uint32)
    aliased = np.zeros(32, dtype=np.uint32)
    addresses = np.zeros(32, dtype=np.uint32)

    module = v2.transpile(pointer_derived_shared_raw_roundtrip)
    result = v2.Engine().run(
        module, {"source": source, "loaded": loaded, "aliased": aliased, "addresses": addresses}
    )

    np.testing.assert_array_equal(result.outputs["loaded"], source)
    np.testing.assert_array_equal(result.outputs["aliased"], source)
    np.testing.assert_array_equal(
        result.outputs["addresses"], np.arange(48, 48 + 32 * 4, 4, dtype=np.uint32)
    )


def test_raw_load_rejects_an_unmapped_integer_when_the_address_is_consumed():
    """Port of ``tests/numsim/integration/test_raw_memory_artifact.py::test_raw_load_rejects_an_unmapped_integer_when_the_address_is_consumed``
    (fail-closed half).

    Dropped: legacy ``analyze(...).unsupported == ()`` (replaced by
    ``v2.transpile`` accepting the kernel), the ``emit_rust_module`` text pin
    ``physical_ptr_from_generic_addresses_u64``, and
    ``match="integer_address_without_binding"``. Asserted: the NumSim run of
    the forged-address load raises ``v2.ExecutionError`` and neither checker
    reports ``clean``. The legacy incomplete verdict/status is
    ``..._is_incomplete`` (v2 gap).
    """

    module = v2.transpile(raw_load_rejects_integer_address)
    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(raw_load_rejects_integer_address, dict(inputs))
        assert report.verdict in ("error", "incomplete"), report.format()
    with pytest.raises(v2.ExecutionError):
        v2.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})


def test_raw_load_rejects_an_unmapped_integer_when_the_address_is_consumed_is_incomplete():
    """Second half of ``tests/numsim/integration/test_raw_memory_artifact.py::test_raw_load_rejects_an_unmapped_integer_when_the_address_is_consumed``.

    Keeps the legacy contract: an integer address with no binding fails
    closed as ``incomplete`` (not an OOB/bad-address proof). Dropped:
    ``"T.ptx.ld" in report.format()`` (legacy report wording).
    """

    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(raw_load_rejects_integer_address, dict(inputs))
        assert report.verdict == "incomplete", report.format()
        assert [finding.kind for finding in report.findings] == ["analysis_incomplete"]
        reasons = [str(finding.details.get("reason")) for finding in report.findings]
        assert len(reasons) == 1 and "integer_address_without_binding" in reasons[0], reasons
    module = v2.transpile(raw_load_rejects_integer_address)
    with pytest.raises(v2.ExecutionError) as caught:
        v2.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})
    stop = _first_stop(caught.value)
    assert stop["status"] == "incomplete", stop
    assert stop["kind"] == "analysis_incomplete", stop
