"""v2 copies of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py``.

The legacy helpers decoded each ``tirx.ptx.tcgen05_mma*`` call with the legacy
registry decoder ``numsim.transpiler.ptx_dialect.decode_ptx_call`` (and
reused ``collector_case`` from ``tests/numsim/runtime/test_tcgen_collectors.py``,
which imports the same decoder at module level). The copies name operands with
:func:`_ptx_operands`, which reads the same TVM PTX table
(``tvm.backend.cuda.ptx.table``: ``TABLE`` keyed by ``op_name``, ``mods``,
``operand_layout``) the legacy decoder wrapped. ``collector_case`` is copied
verbatim for the ``sequence=("",)`` form these tests use.

Other changes: legacy ``run_checked`` / ``assert_rejected`` become v2
synccheck/racecheck verdicts plus ``v2.Engine().run``; the legacy loops are
split into named tests. The GPU oracles keep the legacy
``require_numsim_gpu`` / ``run_gpu_primfunc`` mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm import tirx
from tvm.backend.cuda.ptx.table import TABLE, mods, operand_layout
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_map, structural_walk

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.tcgen_descriptor import INSTR_DESC, encode_dense_instr_descriptor_fields
from tests.numsim.v2.checkers._runnable import assert_clean, requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine

_TABLE_BY_OP_NAME = {entry.op_name: entry for entry in TABLE.values()}


def _ptx_operands(call) -> dict[str, tuple]:
    """Named operand groups of one table-driven ``tirx.ptx.*`` call.

    Arguments are ``operands..., [instruction predicate], modifier tokens...,
    marker``. Only the no-sink calls these kernels build are decoded.
    """

    entry = _TABLE_BY_OP_NAME[str(call.op.name)]
    args = list(call.args)
    metadata = len(entry.slots) + 1
    tokens = tuple(str(node.value) for node in args[-metadata:-1])
    layout = operand_layout(entry, mods(entry, tokens))
    serialized = args[:-metadata]
    lanes = sum(count for _, _, count in layout)
    assert len(serialized) in (lanes, lanes + 1), (str(call.op.name), len(serialized), lanes)
    return {slot.name: tuple(serialized[first : first + count]) for slot, first, count in layout}


def _scalar(operands, name):
    (value,) = operands[name]
    return value


def collector_case(sequence, *, tmem_a=False, ws=False, slot="a", cta_group=1):
    """Verbatim copy of ``tests/numsim/runtime/test_tcgen_collectors.py::collector_case``
    with ``decode_ptx_call`` replaced by :func:`_ptx_operands`."""

    m = 128 * cta_group
    n = 64 if ws else 16 * cta_group
    b_rows = n + 32 if ws else n
    kind = "ti16" if ws else "f16"
    kernel = ti16_kernel(
        False,
        tmem_a,
        cta_group=cta_group,
        m=m,
        ws=ws,
        kind=kind,
        arch="sm_107a" if ws or tmem_a or slot == "b" else "sm_100a",
    )

    def replace_mma(node):
        if not isinstance(node, tirx.Evaluate) or type(node.value).__name__ != "Call":
            return node
        if not str(node.value.op.name).startswith("tirx.ptx.tcgen05_mma"):
            return node
        decoded = _ptx_operands(node.value)
        prefix = f"tcgen05.mma{'.ws' if ws else ''}.cta_group::{cta_group}.kind::{kind}"
        instructions = []
        enabled = 0
        for action in sequence:
            execute = not action.startswith("!")
            action = action.removeprefix("!")
            opcode = prefix + (f".collector::{slot}::{action}" if action else "")
            # The current TVM TS collector-A entry requires explicit collector B.
            if action and tmem_a and not ws and slot == "a":
                opcode += ".collector::b::discard"
            if action and slot == "b" and not ws:
                opcode = prefix + f".collector::a::discard.collector::b::{action}"
            operands = [
                _scalar(decoded, "d_tmem"),
                _scalar(decoded, "a_tmem" if tmem_a else "a_desc"),
                _scalar(decoded, "b_desc"),
                _scalar(decoded, "idesc"),
            ]
            if not ws:
                operands.extend(decoded["disable_output_lane"])
            operands.append(T.ptx.pred(T.uint32(enabled != 0)))
            if ws:
                operands.append(_scalar(decoded, "zero_col_mask"))
            call = T.ptx[opcode](*operands, pred=execute)
            instructions.append(tirx.Evaluate(call))
            enabled += execute
        return instructions[0] if len(instructions) == 1 else tirx.SeqStmt(instructions)

    kernel = kernel.with_body(structural_map(kernel.body, (tirx.Evaluate, replace_mma)))
    a = np.arange(4 * m * 16).reshape(4, m, 16) % 7 - 3
    b = np.arange(b_rows * 16).reshape(b_rows, 16) % 5 - 2
    expected = (a[0].astype(np.float32) @ b[:n].astype(np.float32).T) * sum(
        not action.startswith("!") for action in sequence
    )
    if cta_group == 2:
        expected[129] = 0  # Existing helper's disabled physical output lane.
    encode = (
        (lambda value: (np.abs(value) | ((value < 0).astype(np.int64) << 15)).astype(np.uint16))
        if ws
        else (lambda value: value.astype(np.float16).view(np.uint16))
    )
    expected = expected.astype(np.int32) if ws else expected.astype(np.float32).view(np.int32)
    return (
        kernel,
        {
            "a": encode(a),
            "b": encode(b),
            "metadata": np.zeros((2, 128, 2), np.uint32),
            "zero_mask": np.zeros(1, np.uint64),
            "seed": np.zeros((m, n), np.int32),
            "out": np.zeros((m, n), np.int32),
        },
        expected,
    )


def dispatch_case(
    *,
    bf16=False,
    enabled=True,
    multiple_issuers=False,
    input_descriptor=False,
    descriptor_xor=0,
    tmem_a=False,
):
    kernel, inputs, expected = collector_case(("",), tmem_a=tmem_a)
    kernel = kernel.with_attr("tirx.cuda_arch", "sm_100a")
    metadata = next(parameter for parameter in kernel.params if str(parameter) == "metadata")
    selector = tirx.BufferLoad(metadata, [0, 0, 0])
    if multiple_issuers:
        lanes = []
        structural_walk(
            kernel.body,
            lambda node: (
                lanes.append(node) if type(node).__name__ == "Var" and str(node) == "lane" else None
            ),
        )
        # Distinct types must not split one invalid issue into valid calls.
        selector = (
            tirx.BufferLoad(metadata, [0, lanes[0], 0])
            if input_descriptor
            else T.Cast("uint32", lanes[0])
        )
    predicate = tirx.NE(tirx.BufferLoad(metadata, [0, 0, 1]), T.uint32(0))
    descriptor_bits = 0

    def replace(node):
        nonlocal descriptor_bits
        if isinstance(node, tirx.Evaluate) and type(node.value).__name__ == "Call":
            if not str(node.value.op.name).startswith("tirx.ptx.tcgen05_mma"):
                return node
            decoded = _ptx_operands(node.value)
            descriptor = int(_scalar(decoded, "idesc").value)
            descriptor_bits = descriptor
            return tirx.Evaluate(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                    _scalar(decoded, "d_tmem"),
                    _scalar(decoded, "a_tmem" if tmem_a else "a_desc"),
                    _scalar(decoded, "b_desc"),
                    selector
                    if input_descriptor
                    else T.Select(
                        tirx.NE(selector, T.uint32(0)),
                        T.uint32(descriptor | (1 << 7) | (1 << 10)),
                        T.uint32(descriptor),
                    ),
                    *decoded["disable_output_lane"],
                    T.ptx.pred(T.uint32(0)),
                    pred=predicate,
                )
            )
        if isinstance(node, tirx.IfThenElse):
            mma_sites = []
            structural_walk(
                node.then_case,
                lambda child: (
                    mma_sites.append(child)
                    if type(child).__name__ == "Call"
                    and str(child.op.name).startswith("tirx.ptx.tcgen05_mma")
                    else None
                ),
            )
            if not mma_sites:
                return node

            def select_lane(expression):
                if type(expression).__name__ == "EQ" and str(expression.a) == "lane":
                    return (
                        tirx.LE(expression.a, T.int32(1))
                        if multiple_issuers
                        else tirx.EQ(expression.a, T.int32(7))
                    )
                return expression

            condition = structural_map(node.condition, select_lane)
            return tirx.IfThenElse(condition, node.then_case, node.else_case)
        return node

    kernel = kernel.with_body(structural_map(kernel.body, replace))
    inputs["metadata"][0, 0, :2] = (bf16, enabled)
    if input_descriptor:
        inputs["metadata"][0, 0, 0] = (
            descriptor_bits | (((1 << 7) | (1 << 10)) if bf16 else 0)
        ) ^ descriptor_xor
        if multiple_issuers:
            inputs["metadata"][0, 1, 0] = descriptor_bits | (1 << 7) | (1 << 10)
    if bf16:
        for name in ("a", "b"):
            values = inputs[name].view(np.float16).astype(np.float32)
            inputs[name] = (values.view(np.uint32) >> 16).astype(np.uint16)
    if not enabled:
        expected = inputs["seed"].copy()
    return kernel, inputs, expected


def _copy(inputs):
    return {name: np.copy(value) for name, value in inputs.items()}


def replay(kernel, inputs, expected, *, output="out"):
    for checker in (v2.synccheck, v2.racecheck):
        assert_clean(checker(kernel, _copy(inputs)))
    outputs = v2.Engine().run(v2.transpile(kernel), _copy(inputs)).outputs
    np.testing.assert_array_equal(outputs[output], expected)
    return outputs


def assert_rejected(kernel, inputs, reason):
    """v2 form of legacy ``assert_rejected``: both checkers report an error
    naming ``reason`` and the NumSim run stops with an ``ExecutionError``."""

    for checker in (v2.synccheck, v2.racecheck):
        report = checker(kernel, _copy(inputs))
        assert report.verdict == "error", report.format()
        assert reason in report.format(), report.format()
    with pytest.raises(v2.ExecutionError, match=reason):
        v2.Engine().run(v2.transpile(kernel), _copy(inputs))


_DISPATCH = [
    pytest.param(input_descriptor, bf16, enabled, id=f"{source}-{name}")
    for input_descriptor, source in ((False, "select"), (True, "input"))
    for bf16, enabled, name in (
        (False, True, "f16"),
        (True, True, "bf16"),
        (True, False, "disabled"),
    )
]


@pytest.mark.parametrize(("input_descriptor", "bf16", "enabled"), _DISPATCH)
def test_tcgen_runtime_descriptor_dispatch_outcomes(input_descriptor, bf16, enabled):
    """Port of the dispatch loop of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_runtime_descriptor_dispatch``."""

    replay(*dispatch_case(bf16=bf16, enabled=enabled, input_descriptor=input_descriptor))




@pytest.mark.parametrize("input_descriptor", (False, True), ids=("select", "input"))
def test_tcgen_runtime_descriptor_dispatch_multiple_issuers_fail_closed(input_descriptor):
    """Port of the multiple-issuer rejection of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_runtime_descriptor_dispatch``,
    without the reason text: all three tools stop with an error."""

    kernel, inputs, _ = dispatch_case(multiple_issuers=True, input_descriptor=input_descriptor)
    for checker in (v2.synccheck, v2.racecheck):
        report = checker(kernel, _copy(inputs))
        assert report.verdict == "error", report.format()
    with pytest.raises(v2.ExecutionError):
        v2.Engine().run(v2.transpile(kernel), _copy(inputs))


@pytest.mark.parametrize("input_descriptor", (False, True), ids=("select", "input"))
def test_tcgen_runtime_descriptor_dispatch_rejects_multiple_issuers(input_descriptor):
    """Port of the multiple-issuer rejection of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_runtime_descriptor_dispatch``
    (v2 wording: legacy said "requires exactly one issuing lane")."""

    kernel, inputs, _ = dispatch_case(multiple_issuers=True, input_descriptor=input_descriptor)
    assert_rejected(kernel, inputs, "must be issued by a single thread")


def test_tcgen_runtime_descriptor_signs_stay_runtime_fields():
    """Port of the runtime-field checks of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_runtime_descriptor_dispatch``:
    the descriptor is not a whitelist (negate-A flips the sign), a malformed
    descriptor is ignored when the instruction is disabled, and TMEM-A takes
    an input descriptor."""

    kernel, inputs, _ = dispatch_case(input_descriptor=True, descriptor_xor=1 << 13)
    negated_a = -inputs["a"][0].view(np.float16).astype(np.float32)
    b = inputs["b"].view(np.float16).astype(np.float32)
    replay(kernel, inputs, (negated_a @ b.T).view(np.int32))
    replay(*dispatch_case(input_descriptor=True, enabled=False, descriptor_xor=1 << 6))
    replay(*dispatch_case(input_descriptor=True, tmem_a=True))


@pytest.mark.parametrize(
    ("descriptor_xor", "tmem_a", "message"),
    [
        (1 << 6, False, "descriptor must encode"),
        (1 << 10, False, "mixed"),
        (1 << 15, True, "transpose_a"),
    ],
    ids=("bit6", "mixed-ab", "tmem-transpose-a"),
)
def test_tcgen_runtime_descriptor_rejects_malformed_enabled_descriptor(
    descriptor_xor, tmem_a, message
):
    """Port of the malformed-descriptor rejections of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_runtime_descriptor_dispatch``."""

    kernel, inputs, _ = dispatch_case(
        input_descriptor=True, descriptor_xor=descriptor_xor, tmem_a=tmem_a
    )
    assert_rejected(kernel, inputs, message)


@pytest.mark.numsim_gpu
def test_tcgen_runtime_descriptor_dispatch_gpu_oracle(pytestconfig):
    """Port of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_runtime_descriptor_dispatch_gpu_oracle`` (helpers made legacy-free)."""

    require_numsim_gpu(pytestconfig)
    for input_descriptor in (False, True):
        for bf16, enabled in ((False, True), (True, True), (True, False)):
            kernel, inputs, expected = dispatch_case(
                bf16=bf16, enabled=enabled, input_descriptor=input_descriptor
            )
            gpu = run_gpu_primfunc(kernel, inputs, outputs=("out",), arch="sm_100a")
            np.testing.assert_array_equal(gpu["out"], expected)


def _descriptor_values(kernel, descriptor):
    """The instruction-descriptor words ``kernel`` gives the MMA ``idesc`` operand."""

    if type(descriptor).__name__ == "IntImm":
        return {int(descriptor.value)}
    values = set()

    def collect(node):
        if type(node).__name__ != "Call" or str(node.op.name) != INSTR_DESC:
            return
        destination, *fields = node.args
        if type(descriptor).__name__ != "TensorLoad" or not destination.args[0].source.same_as(
            descriptor.source
        ):
            return
        d_dtype, a_dtype, b_dtype, m, n, k, trans_a, trans_b, cta_group, *flags = (
            field.value for field in fields
        )
        neg_a, neg_b, sat_d, sparse = map(bool, flags)
        values.add(
            encode_dense_instr_descriptor_fields(
                d_dtype=d_dtype,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                m=m,
                n=n,
                k=k,
                trans_a=bool(trans_a),
                trans_b=bool(trans_b),
                cta_group=cta_group,
                neg_a=neg_a,
                neg_b=neg_b,
                sat_d=sat_d,
                sparse=sparse,
            )
        )

    structural_walk(kernel.body, ((Expr, Stmt), collect))
    return values


def with_input_descriptor(kernel):
    parameter = tirx.Var("input_descriptor", "uint32")
    values = set()

    def replace(node):
        if type(node).__name__ != "Call" or not str(node.op.name).startswith(
            "tirx.ptx.tcgen05_mma"
        ):
            return node
        descriptor = _scalar(_ptx_operands(node), "idesc")
        resolved = _descriptor_values(kernel, descriptor)
        assert resolved
        values.update(resolved)
        return type(node)(
            node.op,
            [parameter if arg.same_as(descriptor) else arg for arg in node.args],
            attrs=node.attrs,
            ty_args=node.ty_args,
            span=node.span,
            ret_ty=node.ty,
        )

    body = structural_map(kernel.body, replace)
    assert len(values) == 1
    return (
        tirx.PrimFunc([*kernel.params, parameter], body, kernel.ret_type, kernel.attrs),
        values.pop(),
    )


def input_codec_cases():
    from tests.numsim.microtests.cases.tcgen05_mma_forms import (
        e5m2_layout_f_reference,
        f16_destination_reference,
        make_raw_e4m3_e5m2_arguments,
        make_raw_e5m2_arguments,
        make_raw_f16_destination_arguments,
        raw_e5m2_e4m3_f16_d_ss_m128_layout_d,
        raw_e5m2_ss_m64_layout_f_valid_descriptor,
    )
    from tests.numsim.runtime.test_tcgen05_i8 import i8_case

    kernel, descriptor = with_input_descriptor(raw_e5m2_ss_m64_layout_f_valid_descriptor)
    for make_inputs, a_format, a_dtype in (
        (make_raw_e5m2_arguments, 1, "float8_e5m2"),
        (make_raw_e4m3_e5m2_arguments, 0, "float8_e4m3fn"),
    ):
        inputs = make_inputs()
        inputs["input_descriptor"] = (descriptor & ~(7 << 7)) | (a_format << 7)
        yield (
            kernel,
            inputs,
            e5m2_layout_f_reference(inputs, a_dtype=a_dtype, b_dtype="float8_e5m2"),
        )
    kernel, descriptor = with_input_descriptor(raw_e5m2_e4m3_f16_d_ss_m128_layout_d)
    inputs = make_raw_f16_destination_arguments()
    inputs["input_descriptor"] = descriptor
    yield kernel, inputs, f16_destination_reference(inputs)
    kernel, inputs, expected = i8_case(1, 128, False, False, 1, 0, True, arch="sm_100a")
    kernel, descriptor = with_input_descriptor(kernel)
    inputs["input_descriptor"] = descriptor
    yield kernel, inputs, expected


def test_tcgen_input_descriptor_codecs():
    """Port of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_input_descriptor_codecs``."""

    for kernel, inputs, expected in input_codec_cases():
        replay(kernel, inputs, expected, output="output" if "output" in inputs else "out")
    # The current TVM sparse entry requires SM107 collector syntax; keep the
    # existing model's dynamic-descriptor control off the SM100 GPU path.
    from tests.numsim.runtime.test_tcgen05_sparse_b16 import sparse_float_case

    kernel, inputs, expected = sparse_float_case(1, 64, False, 1, 1, False, False, False)
    kernel, descriptor = with_input_descriptor(kernel)
    inputs["input_descriptor"] = descriptor
    replay(kernel, inputs, expected)


@pytest.mark.numsim_gpu
def test_tcgen_input_descriptor_gpu_oracle(pytestconfig):
    """Port of ``tests/numsim/microtests/test_tcgen_descriptor_dispatch.py::test_tcgen_input_descriptor_gpu_oracle`` (helpers made legacy-free)."""

    require_numsim_gpu(pytestconfig)
    for kernel, inputs, expected in input_codec_cases():
        output = "output" if "output" in inputs else "out"
        gpu = run_gpu_primfunc(kernel, inputs, outputs=(output,), arch="sm_100a")
        np.testing.assert_array_equal(gpu[output], expected)
