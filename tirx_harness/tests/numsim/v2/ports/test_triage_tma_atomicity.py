"""v2 port of ``tests/numsim/runtime/test_tma_atomicity.py::test_tma_atomicity_direction_restrictions``.

Dropped pins:

- ``"T.ptx.cp(" in finding.details["operation"]["source"]["source_text"]``.
Legacy filled ``source_text`` with the TVMScript call text; v2 fills it from
``SiteInfo.text``, which for a call node is the builtin name
(``tirx.ptx.cp_async_bulk_tensor_s2g`` / ``..._g2s``). That is report shape, not
semantics; the copy asserts the call-node identity instead.
- ``reason in report.format()`` for the ``incomplete`` case: v2 puts the
  incomplete reason (``Op(Unsupported): tma_64b_atomicity_load_unmodeled ...``) in
  ``finding.details["reason"]`` and ``format()`` prints the empty message, so the
  copy searches ``report.to_dict()`` (legacy ``assert_rejected`` used ``format()``).

Kept: verdict and message per case in both checkers and NumSim, exactly one
finding, ``operation.source_op_id == operation.source.source_op_id``, and a
positive source line. The two case builders are trimmed copies of the legacy
``atomicity_case`` (atom 32, no flip / im2col / restore) and ``u6_case``
(swizzled load).
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import TensorMap, v2

pytestmark = requires_v2_engine


def atomicity_case(atom, *, store=False, shared_base=0):
    """Trimmed copy of the legacy ``atomicity_case`` (flip/im2col/restore removed)."""
    swizzle = "128B" if atom == 16 else f"128B_ATOM_{atom}B"
    coords = "0, 0"
    values = np.arange(128, dtype=np.uint32) + 0x10000000
    base = np.full(132, 0xA5B6C7D8, np.uint32) if store else values
    inputs = {"values": values, "backing": base} if store else {"output": np.zeros(164, np.uint32)}
    address = f"{shared_base} + i * 4"
    address = f"({address}) ^ (((({address}) >> 7) & {128 // atom - 1}) * {atom})"
    if store:
        body = f"""
    for i in T.serial(128):
        if lane == 0:
            shared[({address}) // 4] = values[i]
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
            T.address_of(descriptor), {coords}, shared.ptr_to([{shared_base // 4}]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
"""
        parameters = 'values: T.Buffer((128,), "uint32"), backing: T.Buffer((132,), "uint32")'
    else:
        body = f"""
    barrier = T.alloc_shared((1,), "uint64")
    if lane == 0:
        for i in T.serial(164):
            shared[i] = T.uint32(0xa5b6c7d8)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 512)
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([{shared_base // 4}]), T.address_of(descriptor), {coords}, barrier.ptr_to([0]))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        for i in T.serial(164):
            output[i] = shared[i]
"""
        parameters = 'output: T.Buffer((164,), "uint32")'
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(descriptor: T.TensorMap(), {parameters}):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_shared((164,), "uint32", align=1024)
    {body}
""",
        {"T": T},
    )
    metadata = dict(
        global_shape=(32, 4),
        global_strides=(128,),
        box_shape=(32, 4),
        element_strides=(1, 1),
        swizzle=swizzle,
        im2col=None,
    )
    return kernel, inputs, base, metadata


def u6_swizzled_load_case(atomicity=32):
    """Trimmed copy of the legacy ``u6_case(swizzle=True, atomicity=...)`` (load, tile)."""
    address = "(i // 3) * 16 + (i % 3) * 4"
    address = f"({address}) ^ (((({address}) >> 7) & {128 // atomicity - 1}) * {atomicity})"
    source = (np.arange(256, dtype=np.uint32) * 37 + 13).astype(np.uint8)
    body = f"""
    shared = T.alloc_shared((68,), "uint32", align=1024)
    barrier = T.alloc_shared((1,), "uint64")
    if lane == 0:
        for i in T.serial(4):
            shared[64 + i] = T.uint32(0xa5b6c7d8)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 192)
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), T.address_of(descriptor), 0, 0, barrier.ptr_to([0]))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        for i in T.serial(48):
            output[i] = shared[({address}) // 4]
        for i in T.serial(4):
            output[48 + i] = shared[64 + i]
"""
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(descriptor: T.TensorMap(), output: T.Buffer((52,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    {body}
""",
        {"T": T},
    )
    storage = np.empty(source.size + 31, np.uint8)
    start = -int(storage.ctypes.data) % 32
    base = storage[start : start + source.size]
    base[:] = source
    metadata = dict(
        global_shape=(128, 2),
        global_strides=(96,),
        box_shape=(128, 2),
        element_strides=(1, 1),
        tma_dtype="uint6",
        swizzle=f"128B_ATOM_{atomicity}B",
        im2col=None,
    )
    return kernel, {"output": np.zeros(52, np.uint32)}, base, metadata


def _assert_rejected(kernel, inputs, reason, verdict):
    reports = [checker(kernel, inputs) for checker in (v2.synccheck, v2.racecheck)]
    for report in reports:
        assert report.verdict == verdict, report.format()
        # v2 keeps an incomplete finding's reason in ``details["reason"]``;
        # ``format()`` prints only the (empty) message, so search the payload.
        assert reason in str(report.to_dict()), report.format()
    with pytest.raises(v2.ExecutionError, match=reason):
        v2.Engine().run(v2.transpile(kernel), inputs)
    return reports


def test_tma_atomicity_direction_restrictions():
    """Port; dropped pin: legacy TVMScript ``source_text`` (see the module docstring)."""
    for store, swizzle, dtype, verdict, message in (
        (True, "128B_ATOM_32B_FLIP_8B", None, "error", "8B flip is only valid for global-to-shared"),
        (False, "128B_ATOM_64B", None, "incomplete", "tma_64b_atomicity_load_unmodeled"),
        (False, "128B_ATOM_64B", "uint6", "error", "64B atomicity loads are invalid for U6 and padded FP4"),
        (False, "128B_ATOM_64B", "fp4", "error", "64B atomicity loads are invalid for U6 and padded FP4"),
    ):
        if dtype is None:
            kernel, inputs, base, metadata = atomicity_case(32, store=store)
        else:
            kernel, inputs, base, metadata = u6_swizzled_load_case(atomicity=32)
            if dtype == "fp4":
                metadata.update(tma_dtype=None, fp4_shared_layout="align16_padded", global_strides=(64,))
        inputs["descriptor"] = TensorMap(base, **{**metadata, "swizzle": swizzle}).numpy()
        for report in _assert_rejected(kernel, inputs, message, verdict):
            assert len(report.findings) == 1, report.format()
            operation = report.findings[0].details["operation"]
            source = operation["source"]
            assert operation["source_op_id"] == source["source_op_id"]
            assert source["kind"] == "ir.Call"
            assert source["source_text"].startswith("tirx.ptx.cp_async_bulk_tensor")
            assert source["source_span"]["line"] > 0
