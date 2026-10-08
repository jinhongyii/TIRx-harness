"""pytest plugin: save every PrimFunc handed to ``numsim.transpile``, then skip.

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    CAPTURE_DIR=/tmp/capture PYTHONPATH=../scripts/numsim-v2 \\
        $PY -m pytest tests/numsim tests/analysis_tools -p capture_plugin -n 48 -q

Each distinct PrimFunc (as given by the caller, before any legacy
normalization) is written to ``$CAPTURE_DIR/<sha1-16>.json`` with
``tvm.ir.save_json``; ``index.<pid>.jsonl`` maps hashes to test node ids.
The tests themselves are skipped right after capture.
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
    from tirx_harness.numsim.transpiler import frontend, host_prelude

    def capture(func):
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
        pytest.skip("captured")

    host_prelude.normalize_host_tensor_map_prelude = capture
    frontend.normalize_host_tensor_map_prelude = capture
