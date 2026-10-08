"""v2 ports of the legacy ``wait_until`` tests that also pinned generated
``module.rust_source`` text (memory-variant sequence, ``declared_wait`` and
``nanosleep`` counts) or the ``Hint:`` text of a report. Those pins are
dropped; outputs, verdicts and finding kinds are kept. The kernel builders are
copied verbatim from the legacy module."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


def _verdicts(kernel, inputs):
    """Each checker's verdict and the shape of what it found (v2 reports)."""

    summary = {}
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(kernel, inputs())
        summary[report.checker_name] = (
            report.verdict,
            sorted((finding.status, finding.kind) for finding in report.findings),
        )
    return summary


# Racecheck delta R3 (docs/development/racecheck-behaviour-deltas.md): a strong
# (acquire) load observing an unordered morally strong write is exempt, and on a
# word that is not a declared ``wait_until`` word it is a ``review`` advisory
# ``UndeclaredProtocolWord``. The raw spin loops below therefore report
# ``review`` where legacy said ``clean`` (rendezvous) or ``error``/data_race
# (packed). The legacy expectation is kept until the test is re-ruled.


def _run(kernel, inputs):
    return v2.Engine().run(v2.transpile(kernel), inputs)


def _rendezvous(*, primitive):
    """A release/acquire rendezvous written in one of the two spellings.

    Warp 0 publishes a payload word and then contributes to the state word;
    warp 1 waits for that contribution and reads the payload behind the
    acquire. One thread alone takes a ticket, so its pre-image is the same
    under every schedule and the kernel's outputs stay exact.
    """

    if primitive:
        publish = (
            'T.ptx.st.release.gpu.global_.s32(slot.ptr_to([0]), T.int32(7))'
        )
        take_ticket = (
            "T.ptx.atom.relaxed.gpu.global_.add.s32("
            'ticket[0], tickets.ptr_to([0]), T.int32(1))'
        )
        signal = 'T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))'
        wait = (
            "T.cuda.wait_until("
            'seen[0], state.ptr_to([0]), seen[0] >= 1, "gpu", "global")'
        )
        read = 'T.ptx.ld.relaxed.gpu.global_.s32(got[0], slot.ptr_to([0]))'
    else:
        publish = 'T.ptx["st.release.gpu.global.s32"](slot.ptr_to([0]), T.int32(7))'
        take_ticket = (
            'T.ptx["atom.relaxed.gpu.global.add.s32"](ticket[0], tickets.ptr_to([0]), T.int32(1))'
        )
        signal = 'T.ptx["red.release.gpu.global.add.s32"](state.ptr_to([0]), T.int32(1))'
        wait = (
            "while seen[0] < 1:\n"
            '                T.ptx["ld.acquire.gpu.global.s32"](seen[0], state.ptr_to([0]))'
        )
        read = 'T.ptx["ld.relaxed.gpu.global.s32"](got[0], slot.ptr_to([0]))'

    return tvm.script.from_source(
        f"""
@T.prim_func
def rendezvous(state: T.Buffer((1,), "int32"), slot: T.Buffer((1,), "int32"),
               tickets: T.Buffer((1,), "int32"), observed: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    got = T.alloc_local((1,), "int32")
    ticket = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            {publish}
            {take_ticket}
            observed[1] = ticket[0]
            {signal}
        else:
            seen[0] = 0
            {wait}
            {read}
            observed[0] = got[0]
""",
        {"T": T},
    )


def _rendezvous_inputs():
    return {
        "state": np.zeros(1, np.int32),
        "slot": np.zeros(1, np.int32),
        "tickets": np.zeros(1, np.int32),
        "observed": np.full(2, -1, np.int32),
    }


# Every width the declared word accepts, with the PTX type each is read as.
_WORD_TYPES = {"int32": "s32", "uint32": "u32", "int64": "s64", "uint64": "u64"}


def _packed(*, primitive, dtype):
    """A wait whose payload rides in the polled word itself.

    The value the wait exits on is the whole message: coherence on the one
    word carries it and no other address is read on the wait's strength. The
    wait still acquires, because there is one lowering and it is cheap enough
    that this shape does not pay for a second one.
    """

    suffix = _WORD_TYPES[dtype]
    if primitive:
        publish = (
            f'T.ptx.st.relaxed.gpu.global_.{suffix}(state.ptr_to([0]), T.{dtype}(7))'
        )
        wait = (
            "T.cuda.wait_until("
            'seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")'
        )
    else:
        load = f"ld.acquire.gpu.global.{suffix}"
        publish = f'T.ptx["st.relaxed.gpu.global.{suffix}"](state.ptr_to([0]), T.{dtype}(7))'
        wait = f'while seen[0] == 0:\n                T.ptx["{load}"](seen[0], state.ptr_to([0]))'

    return tvm.script.from_source(
        f"""
@T.prim_func
def packed(state: T.Buffer((1,), "{dtype}"), observed: T.Buffer((1,), "{dtype}")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "{dtype}")
    if lane == 0:
        if warp == 0:
            {publish}
        else:
            seen[0] = 0
            {wait}
            observed[0] = seen[0]
""",
        {"T": T},
    )


def _packed_inputs(dtype):
    return {
        "state": np.zeros(1, dtype),
        "observed": np.zeros(1, dtype),
    }


def _mixed(*, scoped_reset):
    """One word whose reset is either a scoped access or a plain one.

    A protocol owns its word, and its publisher is spelled in raw PTX, so
    reaching the word without the primitive is what a protocol ordinarily looks
    like. What the word still refuses is a *plain* access that is concurrent
    with the protocol: one carrying neither ordering nor atomicity, which takes
    part in no agreement at all and is the shape that slips past everything
    else -- it is morally strong against nothing and carries no missing edge of
    its own.

    The resetter is a third warp that never waits, so nothing orders it against
    the waiter. A reset by the waiter itself would be ordered by the very edge
    the wait took, and ordered is not a defect -- see
    `test_an_ordered_plain_reset_is_not_reported`.
    """

    reset = (
        "T.ptx.st.relaxed.gpu.global_.s32(state.ptr_to([0]), T.int32(0))"
        if scoped_reset
        else "state[0] = T.int32(0)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def mixed(state: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            T.ptx.st.release.gpu.global_.s32(state.ptr_to([0]), T.int32(7))
        else:
            if warp == 1:
                seen[0] = 0
                T.cuda.wait_until(
                    seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")
                observed[0] = seen[0]
            else:
                {reset}
""",
        {"T": T},
    )


def _bypass_findings(kernel, inputs=None):
    report = v2.racecheck(kernel, inputs if inputs is not None else _packed_inputs("int32"))
    return [finding for finding in report.findings if finding.kind == "signal_protocol_error"]


def _bit_typed(*, primitive):
    """radix's own split: `add` typed, `ld`/`st` bit-typed.

    The migration's whole promise is that the emitted PTX does not change, and
    a kernel that spells its loads `.b32` is the common case (31 of the
    corpus's ordered loads and stores). The declared form has to reach that
    spelling, so this pins it against the raw one.
    """

    if primitive:
        arrive = 'T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))'
        wait = (
            "T.cuda.wait_until("
            'seen[0], state.ptr_to([0]), seen[0] >= 1, "gpu", "global", "b32")'
        )
    else:
        arrive = 'T.ptx["red.release.gpu.global.add.s32"](state.ptr_to([0]), T.int32(1))'
        wait = (
            "while seen[0] < 1:\n"
            '                T.ptx["ld.acquire.gpu.global.b32"](seen[0], state.ptr_to([0]))'
        )

    return tvm.script.from_source(
        f"""
@T.prim_func
def bit_typed(state: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            {arrive}
        else:
            seen[0] = 0
            {wait}
            observed[0] = seen[0]
""",
        {"T": T},
    )


def _backoff_spin(*, primitive):
    """A contended wait, in the two spellings a kernel has for it.

    The hand-written form is what `allgather_gemm` and `gemm_reduce_scatter`
    write: load, test, back off, load again. The primitive form is a single
    `wait_until` carrying `backoff_ns`, which generates the
    same sequence -- the sleep sits inside the loop and ahead of the load, so
    a first poll that succeeds pays nothing.
    """

    if primitive:
        spin = (
            "T.cuda.wait_until(\n"
            "                seen[0], state.ptr_to([0]), seen[0] >= 1,\n"
            '                "gpu", "global", backoff_ns=40)'
        )
        publish = (
            'T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))'
        )
    else:
        spin = (
            'T.ptx["ld.acquire.gpu.global.s32"](seen[0], state.ptr_to([0]))\n'
            "            while seen[0] < 1:\n"
            "                T.cuda.nano_sleep(T.uint64(40))\n"
            '                T.ptx["ld.acquire.gpu.global.s32"](seen[0], state.ptr_to([0]))'
        )
        publish = 'T.ptx["red.release.gpu.global.add.s32"](state.ptr_to([0]), T.int32(1))'

    return tvm.script.from_source(
        f"""
@T.prim_func
def backoff(state: T.Buffer((1,), "int32"), slot: T.Buffer((1,), "int32"),
            observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            slot[0] = 7
            {publish}
        else:
            seen[0] = 0
            {spin}
            observed[0] = slot[0]
""",
        {"T": T},
    )


def _backoff_inputs():
    return {
        "state": np.zeros(1, np.int32),
        "slot": np.zeros(1, np.int32),
        "observed": np.full(1, -1, np.int32),
    }


def test_rendezvous_matches_raw_spelling():
    """Port of ``tests/numsim/runtime/test_wait_until.py::test_rendezvous_matches_raw_spelling``.

    Dropped pins: ``_memory_variants(module.rust_source) ==
    _memory_variants(raw_module.rust_source)`` and the two
    ``_declared_wait_count(rust_source)`` counts (legacy generated Rust).
    The raw-spelling racecheck verdict lives in
    :func:`test_rendezvous_raw_spelling_racecheck_is_undeclared_word_review` (delta R3).
    """

    raw = _rendezvous(primitive=False)
    primitive = _rendezvous(primitive=True)
    raw_outputs = _run(raw, _rendezvous_inputs()).outputs
    outputs = _run(primitive, _rendezvous_inputs()).outputs

    for name in _rendezvous_inputs():
        np.testing.assert_array_equal(outputs[name], raw_outputs[name])
    assert _verdicts(primitive, _rendezvous_inputs)["racecheck"] == ("clean", [])
    np.testing.assert_array_equal(outputs["observed"], np.array([7, 0], np.int32))


def test_rendezvous_raw_spelling_racecheck_is_undeclared_word_review():
    """Raw-spelling verdict half of ``tests/numsim/runtime/test_wait_until.py::test_rendezvous_matches_raw_spelling``.

    Delta R3: legacy ``("clean", [])``; the raw ``ld.acquire`` spin on an
    undeclared word is now a ``review`` ``UndeclaredProtocolWord`` advisory."""

    assert _verdicts(_rendezvous(primitive=False), _rendezvous_inputs)["racecheck"] == (
        "review",
        [("review", "undeclared_protocol_word")],
    )


@pytest.mark.parametrize("dtype", sorted(_WORD_TYPES))
def test_packed_wait_matches_raw_spelling(dtype):
    """Port of ``tests/numsim/runtime/test_wait_until.py::test_packed_wait_matches_raw_spelling``.

    Dropped pin: ``_memory_variants(module.rust_source) ==
    _memory_variants(raw_module.rust_source)``. The raw-spelling racecheck
    verdict lives in :func:`test_packed_wait_raw_spelling_racecheck_is_undeclared_word_review`
    (delta R3).
    """

    raw = _packed(primitive=False, dtype=dtype)
    primitive = _packed(primitive=True, dtype=dtype)
    inputs = lambda: _packed_inputs(dtype)  # noqa: E731
    raw_outputs = _run(raw, inputs()).outputs
    outputs = _run(primitive, inputs()).outputs

    for name in inputs():
        np.testing.assert_array_equal(outputs[name], raw_outputs[name])
    assert _verdicts(primitive, inputs)["racecheck"] == ("clean", [])
    np.testing.assert_array_equal(outputs["observed"], np.array([7], dtype))


@pytest.mark.parametrize("dtype", sorted(_WORD_TYPES))
def test_packed_wait_raw_spelling_racecheck_is_undeclared_word_review(dtype):
    """Raw-spelling verdict half of ``tests/numsim/runtime/test_wait_until.py::test_packed_wait_matches_raw_spelling``.

    Delta R3: legacy ``("error", [data_race])`` came from the global shadow never
    exempting a strong generic load against a morally strong store; that pair is
    now exempt, and polling the undeclared word is a ``review``
    ``UndeclaredProtocolWord`` advisory."""

    inputs = lambda: _packed_inputs(dtype)  # noqa: E731
    assert _verdicts(_packed(primitive=False, dtype=dtype), inputs)["racecheck"] == (
        "review",
        [("review", "undeclared_protocol_word")],
    )


def test_a_plain_access_on_a_declared_word_is_reported():
    """Port of ``tests/numsim/runtime/test_wait_until.py::test_a_plain_access_on_a_declared_word_is_reported``.

    Dropped pin: ``"Hint:" in racecheck(kernel, ...).format()`` (legacy
    report hint text).
    """

    kernel = _mixed(scoped_reset=False)
    findings = _bypass_findings(kernel)
    assert findings, "a plain store on a claimed word must be reported"
    assert all(finding.status == "error" for finding in findings)
    assert all("category" not in finding.details for finding in findings)
    assert all(finding.details["kind"] == "signal_protocol_error" for finding in findings)
    assert "without a happens-before relationship" in findings[0].message


def test_bit_typed_word_matches_raw_spelling():
    """Port of ``tests/numsim/runtime/test_wait_until.py::test_bit_typed_word_matches_raw_spelling``.

    Dropped pin: ``_memory_variants(module.rust_source) ==
    _memory_variants(raw_module.rust_source)``.
    """

    raw = _bit_typed(primitive=False)
    primitive = _bit_typed(primitive=True)

    raw_outputs = _run(raw, _packed_inputs("int32")).outputs
    outputs = _run(primitive, _packed_inputs("int32")).outputs
    np.testing.assert_array_equal(outputs["observed"], raw_outputs["observed"])
    np.testing.assert_array_equal(outputs["observed"], np.array([1], np.int32))


def test_a_backoff_is_the_cuda_loops_business_and_not_the_engines():
    """Port of ``tests/numsim/runtime/test_wait_until.py::test_a_backoff_is_the_cuda_loops_business_and_not_the_engines``.

    Dropped pins: ``"control::nanosleep" not in module.rust_source`` and
    ``raw.rust_source.count("control::nanosleep") == 1``.
    """

    outputs = _run(_backoff_spin(primitive=True), _backoff_inputs()).outputs
    np.testing.assert_array_equal(outputs["observed"], np.array([7], np.int32))
    v2.racecheck(_backoff_spin(primitive=True), _backoff_inputs()).require_clean()
