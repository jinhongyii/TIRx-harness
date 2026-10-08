"""Racecheck delta T18 (docs/development/racecheck-behaviour-deltas.md): a
one-shot raw read of a declared protocol word made before its publication.

Legacy reported #677's "read once, no retry" shape. v2 reports only a raw
read that observed an unordered write; a read delivered before the
publication cannot be told apart from a spin loop's first iteration without
loop information, so v2 reports nothing (coordinator ruling: documented
limitation). Both copies are STRICT xfails asserting the legacy finding, so
they flip (XPASS -> failure) if the limitation is ever lifted. Kernels are
copied verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_T18 = pytest.mark.xfail(
    strict=True,
    reason=(
        "T18 documented limitation (racecheck-behaviour-deltas.md row T18; expected outcome, "
        "category_overrides.tsv category=expected): a raw read "
        "of a declared word before its publication is indistinguishable from a spin's first "
        "iteration without loop information; v2 reports clean"
    ),
)

GEN = 1
WORD = (GEN << 32) | 0xDEADBEEF


@T.prim_func
def read_once(slot: T.Buffer((1,), "uint64"), sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(0))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD))
        else:
            observed = T.alloc_local((1,), "uint64")
            T.ptx.ld.relaxed.cluster.global_.u64(observed[0], slot.ptr_to([0]))
            sink[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))


def _u64(n=1):
    return np.zeros(n, np.uint64)


@_T18
def test_declared_word_shape_read_once_no_retry():
    """Copy of the ``read_once_no_retry`` parametrization of
    ``tests/analysis_tools/racecheck/test_declared_word_shapes.py::test_declared_word_shape_gets_the_verdict_it_earns``
    (legacy: verdict ``error`` with access pairs ``("write_read",)``). The
    other parametrizations are covered by the Rust ports (racecheck.tsv)."""

    report = v2.racecheck(read_once, {"slot": _u64(), "sink": _u64()})
    assert report.verdict == "error", report.format()
    assert sorted({f.details.get("access_pair") for f in report.findings}) == ["write_read"], report.format()


def _packed_payload(*, retry):
    """Copy of ``_packed_payload`` in ``tests/numsim/runtime/test_wait_until.py``."""

    read = (
        "T.cuda.wait_until(\n"
        "                observed[0], slot.ptr_to([0]),\n"
        '                T.Cast("uint32", T.shift_right(observed[0], T.uint64(32)))'
        " == T.uint32(1),\n"
        '                "gpu", "global")'
        if retry
        else (
            "T.ptx.ld.relaxed.gpu.global_.u64(\n"
            '                observed[0], slot.ptr_to([0]))'
        )
    )
    word = (1 << 32) | 0xDEADBEEF
    return tvm.script.from_source(
        f"""
@T.prim_func
def packed(slot: T.Buffer((1,), "uint64"), sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    observed = T.alloc_local((1,), "uint64")
    if lane == 0:
        if cta == 0:
            T.ptx.st.release.gpu.global_.u64(slot.ptr_to([0]), T.uint64({word}))
        else:
            observed[0] = T.uint64(0)
            {read}
            sink[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))
""",
        {"T": T},
    )


def _packed_payload_inputs():
    return {"slot": np.zeros(1, np.uint64), "sink": np.zeros(1, np.uint64)}


@_T18
def test_reading_the_word_once_is_not_made_correct_by_declaring_it():
    """Port of ``tests/numsim/runtime/test_wait_until.py::test_reading_the_word_once_is_not_made_correct_by_declaring_it``
    (legacy: any non-clean verdict)."""

    report = v2.racecheck(_packed_payload(retry=False), _packed_payload_inputs())
    assert report.verdict != "clean", report.format()
