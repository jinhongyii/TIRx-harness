"""v2 port of the legacy FP32 atomic ``.noftz`` test; the
``module.rust_source`` variant pin is dropped."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2

pytestmark = requires_v2_engine


ATOMIC_CASES = [
    (kind, width, space)
    for kind in ("atom", "sink", "red")
    for width, space in ((1, "global"), (1, "shared::cta"), (2, "global"), (4, ""))
]


def atomic_kernel(kind, width, space, *, noftz=True, offset=0, race=False):
    mnemonic = "red" if kind == "red" else "atom"
    tokens = [mnemonic, "relaxed", "cta", space, "add", "noftz" if noftz else ""]
    tokens += [f"v{width}" if width > 1 else "", "f32"]
    spelling = ".".join(token for token in tokens if token)
    pointer = "shared" if space.startswith("shared") else "destination"
    arguments = [f"{pointer}.ptr_to([lane * {width} + {offset}])"]
    values = [f"value[lane * {width} + {i}]" for i in range(width)]
    if kind == "atom":
        arguments.insert(0, ", ".join(f"old[{i}]" for i in range(width)))
    arguments.extend(values)
    if width > 1:
        arguments.append("pred=lane % 2 == 0")
        if kind == "atom":
            arguments.append("preserve_dst=True")
    return tvm.script.from_source(
        f"""
@T.prim_func
def atomic(destination: T.Buffer((128,), "float32"), value: T.Buffer((128,), "float32"),
           returned: T.Buffer((128,), "float32")):
    T.device_entry()
    warp = T.warp_id([{2 if race else 1}])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared", align=16)
    old = T.alloc_local(({width},), "float32")
    if warp == 0:
        for i in T.serial({width}):
            shared[lane * {width} + i] = destination[lane * {width} + i]
            old[i] = T.float32(-1)
        T.ptx["{spelling}"]({", ".join(arguments)})
        for i in T.serial({width}):
            returned[lane * {width} + i] = old[i]
            {"destination[lane * " + str(width) + " + i] = shared[lane * " + str(width) + " + i]" if pointer == "shared" else "T.evaluate(0)"}
    else:
        destination[lane * {width}] = T.float32(3)
""",
        {"T": T},
    )


def atomic_inputs():
    # Tiny inputs; normal cancellation to a subnormal; tie-to-even; signed
    # zero; infinities and NaNs. Old-value payloads must remain bit-exact.
    left = [1, 0x80000001, 0x00800001, 0x3F800000, 0x80000000, 0x7F800000, 0x7FC12345, 0x00800000]
    right = [1, 0x80000001, 0x80800000, 0x33800000, 0x80000000, 0xFF800000, 0x3F800000, 0x807FFFFF]
    return {
        # An odd period ensures predicated v2/v4 lanes exercise every pair.
        "destination": np.resize(np.array([*left, 0x3F800001], np.uint32), 128).view(np.float32),
        "value": np.resize(np.array([*right, 0x33800000], np.uint32), 128).view(np.float32),
        "returned": np.full(128, -1, np.float32),
    }


def assert_float_bits(actual, expected):
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan)
    np.testing.assert_array_equal(actual[~nan].view(np.uint32), expected[~nan].view(np.uint32))


def addition_expected(*, flush=False):
    # Hand-derived RNE results, independent of both the engine and the host's
    # floating-point flush mode. NaN payloads are intentionally unspecified.
    bits = np.array(
        [2, 0x80000002, 1, 0x3F800000, 0x80000000, 0x7FC00000, 0x7FC00000, 1, 0x3F800002],
        np.uint32,
    )
    if flush:
        bits[[0, 1, 2, 7]] = [0, 0x80000000, 0, 0x00800000]
    return np.resize(bits, 128).view(np.float32)


def _cases():
    for kind, width, space in ATOMIC_CASES:
        yield pytest.param(kind, width, space, id=f"{kind}-{width}-{space}")


@pytest.mark.parametrize("kind,width,space", list(_cases()))
def test_atomic_f32_noftz(kind, width, space):
    """Port of ``tests/numsim/runtime/test_atomic_f32_noftz.py::test_atomic_f32_noftz``.

    Dropped pin: ``("variant::Add<true>" in module.rust_source) is noftz``
    (legacy generated-Rust text). The ``.noftz`` semantics stay asserted
    through the exact result bits (flush vs no-flush expectations).
    """
    for noftz in (False, True):
        kernel = atomic_kernel(kind, width, space, noftz=noftz)
        inputs = atomic_inputs()
        for checker in (v2.synccheck, v2.racecheck):
            checker(kernel, {name: value.copy() for name, value in inputs.items()}).require_clean()
        module = v2.transpile(kernel)
        result = v2.Engine().run(module, {name: value.copy() for name, value in inputs.items()})
        selected = np.arange(128) < 32 * width
        if width > 1:
            selected &= np.arange(128) // width % 2 == 0
        flush = not noftz and not space.startswith("shared")
        left = inputs["destination"]
        expected = left.copy()
        expected[selected] = addition_expected(flush=flush)[selected]
        assert_float_bits(result.outputs["destination"], expected)
        returned = inputs["returned"].copy()
        if kind == "atom":
            returned[selected] = left[selected]
        np.testing.assert_array_equal(
            result.outputs["returned"].view(np.uint32), returned.view(np.uint32)
        )
