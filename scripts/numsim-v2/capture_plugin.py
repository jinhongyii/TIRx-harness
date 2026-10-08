"""pytest plugin: save every PrimFunc handed to ``v2.transpile``, then skip.

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    NUMSIM_IMPL=v2 CAPTURE_DIR=/tmp/capture PYTHONPATH=../scripts/numsim-v2 \\
        $PY -m pytest tests/numsim tests/analysis_tools -p capture_plugin -n 48 -q

``NUMSIM_IMPL=v2`` routes the public ``numsim.transpile`` / ``racecheck`` /
``synccheck`` names to v2 (``tests/conftest.py``). Until the legacy layer is
deleted, kernels reached only through legacy internals are also captured at
the legacy normalization hook (same file format). Every PrimFunc reaching
``tirx_harness.numsim.v2.compile.transpile`` (directly, or through the v2
checkers), as given by the caller, is written to
``$CAPTURE_DIR/<sha1-16>.json`` with ``tvm.ir.save_json``; a sequence of launches
is captured one PrimFunc per file. ``index.<pid>.jsonl`` maps hashes to test
node ids. The tests themselves are skipped right after capture. The format is
the one the legacy-hooked plugin wrote, so ``lower_sweep.py`` and
``inventory.py`` read either.
"""

import hashlib
import json
import os

import pytest

OUT = os.environ.get("CAPTURE_DIR")
_current = {"nodeid": None}


def pytest_runtest_setup(item):
    _current["nodeid"] = item.nodeid


def pytest_configure(config):
    if not OUT:
        raise pytest.UsageError("capture_plugin needs CAPTURE_DIR")
    os.makedirs(OUT, exist_ok=True)
    import tvm
    from tirx_harness.numsim.v2 import compile as v2_compile

    def record(func):
        text = tvm.ir.save_json(func)
        digest = hashlib.sha1(text.encode()).hexdigest()[:16]
        path = os.path.join(OUT, digest + ".json")
        if not os.path.exists(path):
            tmp = f"{path}.tmp{os.getpid()}"
            with open(tmp, "w") as handle:
                handle.write(text)
            os.replace(tmp, path)
        attrs = func.attrs
        has_name = attrs is not None and "global_symbol" in attrs
        name = str(attrs["global_symbol"]) if has_name else None
        row = {"hash": digest, "nodeid": _current["nodeid"], "name": name}
        with open(os.path.join(OUT, f"index.{os.getpid()}.jsonl"), "a") as handle:
            handle.write(json.dumps(row) + "\n")

    original_functions = v2_compile._functions

    def capture(func):
        for one in original_functions(func):
            record(one)
        pytest.skip("captured")

    # `transpile` resolves `_functions` from its module globals on every call,
    # before the module cache lookup, so cache hits are captured too.
    v2_compile._functions = capture

    # Until step 5, tests that reach a kernel only through legacy internals
    # (`analyze`, `emit_rust_module`, the legacy `transpile`) are captured at the
    # legacy normalization hook too, so the capture set stays the one
    # lowering-inventory.md Part F was measured on. The legacy modules are
    # deleted at step 5 and this block then does nothing.
    try:
        from tirx_harness.numsim.transpiler import frontend, host_prelude
    except ImportError:
        return

    def capture_one(func):
        record(func)
        pytest.skip("captured")

    host_prelude.normalize_host_tensor_map_prelude = capture_one
    frontend.normalize_host_tensor_map_prelude = capture_one
