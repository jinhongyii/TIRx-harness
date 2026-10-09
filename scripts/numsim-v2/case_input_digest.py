"""Print a sha256 per prepared input array of canonical corpus cases, and the
engine's bound-input digest, to compare hosts (CI vs a dev host).

Usage (from ``tirx_harness/``)::

    python ../scripts/numsim-v2/case_input_digest.py [CASE ...]   # default: flash_attention_backward_sm100

Read-only and fast: it prepares each case exactly as ``tests/conformance``
does (``entry.prepare()``), computes the reference (as the conformance run
does before executing), lowers the kernel and binds the inputs, but runs no
kernel. If the per-input hashes match across hosts while the conformance
output hashes differ, the divergence is in the engine's numerics on that CPU;
if they differ, it is in the (CPU-dispatched) input generation.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES  # noqa: E402

from tirx_harness import numsim  # noqa: E402
from tirx_harness.numsim.v2.run import canonicalize_inputs  # noqa: E402

DEFAULT_CASES = ("flash_attention_backward_sm100",)


def digest(value) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        array = np.ascontiguousarray(np.asarray(value))
        data = array.view(np.uint8).tobytes() if array.dtype != object else repr(value).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def describe(name: str, value) -> str:
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        array = np.asarray(value)
        text = digest(array)
        if getattr(value, "_tensor_map_base", None) is not None:
            # A host tensor-map image: bytes [0, 8) hold the host address.
            text = digest(array.view(np.uint8).tobytes()[8:]) + " (tensor map, address masked)"
        return f"  input {name:<28} {str(array.dtype):<10} {str(tuple(array.shape)):<20} {text}"
    return f"  input {name:<28} scalar     {value!r:<20} {digest(np.asarray(value))}"


def main(argv: list[str]) -> int:
    names = argv or list(DEFAULT_CASES)
    for name in names:
        entry = next(case for case in CANONICAL_KERNEL_CASES if case.name == name)
        case = entry.prepare()
        args = dict(case.args)
        print(f"case {name}")
        for key in sorted(args):
            print(describe(key, args[key]))
        expected = copy.deepcopy(case.reference())
        for key in sorted(expected):
            print(f"  ref   {key:<28} {digest(expected[key])}")
        module = numsim.transpile(case.kernel)
        bound = canonicalize_inputs(module, case.args)
        engine = hashlib.sha256()
        for key in sorted(bound):
            native = bound[key].native
            if native[0] == "buffer":
                part = digest(native[1])
            elif native[0] == "tensor_map_of":
                # Descriptor image minus its first 8 bytes (the host global
                # address, which differs per process); the engine re-bases it.
                part = f"{native[1]}+{native[2]}:{digest(bytes(native[3])[8:])}"
            else:
                part = repr(native)
            engine.update(f"{key}={part};".encode())
        print(f"  engine bound-input digest {engine.hexdigest()[:16]} ({len(bound)} bindings)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
