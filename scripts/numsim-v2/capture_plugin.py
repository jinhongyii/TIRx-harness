"""pytest plugin: save every PrimFunc handed to ``v2.transpile``, then skip.

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    CAPTURE_DIR=/tmp/capture PYTHONPATH=../scripts/numsim-v2 \\
        $PY -m pytest tests/numsim tests/analysis_tools -p capture_plugin -n 48 -q

Every PrimFunc reaching
``tirx_harness.numsim.v2.compile.transpile`` (directly, or through the v2
checkers), as given by the caller, is written to
``$CAPTURE_DIR/<sha1-16>.json`` with ``tvm.ir.save_json``; a sequence of launches
is captured one PrimFunc per file. ``index.<pid>.jsonl`` maps hashes to test
node ids. The tests themselves are skipped right after capture.
``lower_sweep.py`` and ``inventory.py`` read this format.
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
