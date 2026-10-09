"""Corpus kernel inputs must not depend on the host's BLAS kernel or ISA dispatch.

`flash_attention_backward_sm100`'s `LSE_g`/`dpsum_g` inputs were computed with
float32 `np.matmul`, which OpenBLAS runs with a per-CPU kernel; the inputs, and
so every output hash, differed between the CI runner and the dev host. The
case now computes them without BLAS. This test prepares the inputs in two
processes, one forced onto a different OpenBLAS kernel, and requires identical
bytes. It is a positive control: on a host where the BLAS kernel matters, the
old float32-matmul inputs fail it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SCRIPT = """
import hashlib, sys
import numpy as np
from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES
entry = next(case for case in CANONICAL_KERNEL_CASES if case.name == "flash_attention_backward_sm100")
args = entry.prepare().args
for name in ("Q_g", "K_g", "V_g", "dO_g", "LSE_g", "dpsum_g"):
    data = np.ascontiguousarray(np.asarray(args[name])).view(np.uint8).tobytes()
    print(name, hashlib.sha256(data).hexdigest())
"""


def _launch(extra_env: dict[str, str]) -> subprocess.Popen[str]:
    env = dict(os.environ, **extra_env)
    root = Path(__file__).resolve().parents[3]
    return subprocess.Popen(
        [sys.executable, "-c", _SCRIPT],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_flash_attention_backward_inputs_do_not_depend_on_the_blas_kernel() -> None:
    # `Sandybridge` is an OpenBLAS core every x86-64 host can run whose sgemm
    # rounds differently from the Haswell/Zen/SkylakeX kernels.
    runs = [_launch({}), _launch({"OPENBLAS_CORETYPE": "Sandybridge"})]
    outputs = []
    for run in runs:
        stdout, stderr = run.communicate(timeout=600)
        assert run.returncode == 0, stderr
        outputs.append(stdout)
    assert outputs[0] == outputs[1], f"default:\n{outputs[0]}\nSandybridge:\n{outputs[1]}"
