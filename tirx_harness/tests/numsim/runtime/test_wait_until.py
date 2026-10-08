"""The declared synchronization word lowers to exactly its raw spelling.

``tirx.cuda.wait_until`` emits the poll loop a kernel writes by hand. These
tests write the same protocol twice — once through the primitive, once through
the raw spelling it stands for — and require the two to agree on the numerical
result and on the checker verdict. Nothing about the declaration may change
what a kernel does; it only lets an analysis tell a protocol's own accesses
from a stray one.
"""

import re

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.support.execution import run_checked
from tests.numsim.support.wait_until import indexed_predicate_case, initial_case


def _memory_variants(rust_source):
    """The memory instructions the engine will execute, in order.

    The wait is dropped from both spellings, because the two express it
    differently on purpose. Written raw it is a loop of loads, and the engine
    has to execute every one of them; written through the primitive it is one
    suspending operation and the engine does the waiting itself. That is the
    whole point of the operation — a spin's reads all race the publisher's
    write by construction, so handing them to the memory model reports every
    correct protocol. The CUDA both spellings lower to is still the same loop,
    instruction for instruction; `tests/python/tirx/codegen/test_cuda_wait_until.py`
    is where that is pinned.

    So: `mem::declared_wait` goes, and so does everything inside a native
    `while` body. What the two spellings have to agree on is everything else.
    """

    executed = re.sub(r"v2::mem::declared_wait::<.*?\);\n", "", rust_source, flags=re.S)
    executed = re.sub(
        r"v2::control::while_enter\(.*?v2::control::while_exit\(.*?;\n",
        "",
        executed,
        flags=re.S,
    )
    variants = []
    for call in re.finditer(r"v2::mem::(?:ld|st|atom|red)::<(.*?)>>\(", executed, flags=re.S):
        spelling = call.group(1)
        # A thread-local scalar is bookkeeping, not one of the protocol's
        # instructions: where the wait leaves its exit value is the caller's
        # business and the two spellings put it in different places.
        if "v2::Local" in spelling:
            continue
        variants.extend(re.findall(r"v2::mem::variant::(\w+)", spelling))
    return variants


def _declared_wait_count(rust_source):
    return rust_source.count("v2::mem::declared_wait::<")


def _verdicts(kernel, inputs):
    """Each checker's verdict and the shape of what it found."""

    summary = {}
    for checker in (synccheck, racecheck):
        report = checker(kernel, inputs())
        summary[report.checker_name] = (
            report.verdict,
            sorted((finding.status, finding.kind) for finding in report.findings),
        )
    return summary


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


@pytest.mark.parametrize("dtype", sorted(_WORD_TYPES))
@pytest.mark.parametrize("predicate_op", ["Select", "if_then_else"])
def test_initial_wait_rechecks_conditional_predicate(dtype, predicate_op, tmp_path):
    case = initial_case(dtype, predicate_op=predicate_op)
    result = run_checked(case.kernel, case.args, outputs=case.outputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], case.reference()["out"])


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


def test_declared_rendezvous_is_clean():
    kernel = _rendezvous(primitive=True)
    for checker in (synccheck, racecheck):
        checker(kernel, _rendezvous_inputs()).require_clean()


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


def _ordered_reset():
    """The reset a real kernel writes: plain, but behind a rendezvous.

    A workspace counter cleared with a plain store between two grid-wide
    barriers is ordinary, correct code -- every wait on the word is ordered
    against the clear. The claim on the address must not turn that into a
    finding, or the checker reports an error on every kernel that reuses its
    workspace.

    The rendezvous here is a second word, published and waited on through the
    primitive, so the order is one the program has and not one the reset's own
    word handed out.
    """

    return tvm.script.from_source(
        """
@T.prim_func
def ordered_reset(
    state: T.Buffer((1,), "int32"),
    gate: T.Buffer((1,), "int32"),
    observed: T.Buffer((1,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    ready = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            state[0] = T.int32(0)
            T.ptx.st.release.gpu.global_.s32(gate.ptr_to([0]), T.int32(1))
            T.ptx.st.release.gpu.global_.s32(state.ptr_to([0]), T.int32(7))
        else:
            ready[0] = 0
            T.cuda.wait_until(
                ready[0], gate.ptr_to([0]), ready[0] != 0, "gpu", "global")
            seen[0] = 0
            T.cuda.wait_until(
                seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")
            observed[0] = seen[0]
""",
        {"T": T},
    )


def _bypass_findings(kernel, inputs=None):
    report = racecheck(kernel, inputs if inputs is not None else _packed_inputs("int32"))
    return [finding for finding in report.findings if finding.kind == "signal_protocol_error"]


def test_a_scoped_reset_is_not_reported(tmp_path):
    # The positive control's twin: the identical protocol whose reset carries a
    # scope keeps the word clean, so the finding above is caused by the plain
    # access and not by the protocol's own shape.
    kernel = _mixed(scoped_reset=True)
    assert not _bypass_findings(kernel)


def test_an_ordered_plain_reset_is_not_reported(tmp_path):
    # The other twin: the same plain store, now ordered against every wait on
    # the word by a rendezvous the program already had. Being plain is not the
    # defect -- being concurrent with the protocol is -- so this one is clean.
    inputs = {
        "state": np.zeros(1, "int32"),
        "gate": np.zeros(1, "int32"),
        "observed": np.zeros(1, "int32"),
    }
    assert not _bypass_findings(_ordered_reset(), inputs)


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


def _two_arrivals(*, target):
    """Two publishers, one waiter, and a target that decides what it may read.

    Warp 0 publishes `first` and arrives; warp 1 publishes `second` and
    arrives; warp 2 waits for the counter to reach `target` and then reads
    `second`. Only the second arrival releases `second`, so reading it is
    ordered exactly when the wait cannot have left before both arrived.

    This is what separates the declared edge from the read-from edge the
    engine already had. Read-from hands the waiter whatever version its load
    happened to observe -- at `target=1` that can still be the second arrival,
    which would make the defect disappear on the schedule that runs. API §3
    takes the *earliest* write the predicate accepts instead, so `target=1`
    is judged on the first arrival alone however late the loop actually left.
    """

    return tvm.script.from_source(
        f"""
@T.prim_func
def two_arrivals(state: T.Buffer((1,), "int32"), first: T.Buffer((1,), "int32"),
                 second: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            first[0] = 11
            T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))
        else:
            if warp == 1:
                second[0] = 22
                T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))
            else:
                seen[0] = 0
                T.cuda.wait_until(
                    seen[0], state.ptr_to([0]), seen[0] >= {target}, "gpu", "global")
                observed[0] = second[0]
""",
        {"T": T},
    )


def _two_arrivals_inputs():
    return {
        "state": np.zeros(1, np.int32),
        "first": np.zeros(1, np.int32),
        "second": np.zeros(1, np.int32),
        "observed": np.zeros(1, np.int32),
    }


def _woken_by_a_bypass():
    """A wait let out by a plain write.

    The two rules meet here. A scoped write is how a protocol publishes now, so
    it is not the defect; a *plain* one is. The address claim reports it, and
    the wait reports separately that nothing in the word's history explains the
    value it left on, so it built no edge. Both are needed: the claim names the
    defect, and the wait says the reader is holding an ordering it was never
    given.
    """

    return tvm.script.from_source(
        """
@T.prim_func
def woken_by_a_bypass(state: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            state[0] = T.int32(7)
        else:
            seen[0] = 0
            T.cuda.wait_until(
                seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")
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


def _packed_payload(*, retry):
    """#677's examples 1 and 2, written through the primitive.

    The payload rides in the polled word itself -- the generation in the high
    half, the value in the low half -- which is the shape DeepEP's notify slot
    uses and the one a separate payload buffer cannot stand in for. Both
    versions declare the word; they differ only in whether the reader waits.

    A wait that retries is the protocol working. A single load is #677's "read
    once, no retry": the reader takes whatever the word held, and declaring it
    must not make that look correct.
    """

    read = (
        "T.cuda.wait_until(\n"
        "                observed[0], slot.ptr_to([0]),\n"
        '                T.Cast("uint32", T.shift_right(observed[0], T.uint64(32)))'
        " == T.uint32(1),\n"
        '                "gpu", "global")'
        if retry
        else (
            # Relaxed, not acquiring. An acquiring single read is what an
            # acquiring spin is made of, and the checker sees accesses rather
            # than loops, so it cannot separate the two: sparing the loop
            # spares this. A relaxed read takes the value and no order, which
            # is the shape the claim can name on every schedule.
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


