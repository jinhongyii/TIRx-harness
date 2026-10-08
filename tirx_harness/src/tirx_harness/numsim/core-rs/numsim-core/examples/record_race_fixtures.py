"""Record the racecheck bench fixtures (module, bound inputs, run config) of corpus cases.

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    $PY src/tirx_harness/numsim/core-rs/numsim-core/examples/record_race_fixtures.py OUT_DIR [CASE ...]

A case is a canonical corpus case name, or ``mega_moe:<config label>`` for the
``sm100_fp8_fp4_mega_moe`` racecheck performance configs (148 SMs, 16 workers, as in
``tests/analysis_tools/racecheck/corpus/test_canonical_kernels_racecheck.py``;
``t64_h2048_i1536_e96_k4_g1`` is that test's medium config). The fixture files are
named after the case, with ``:`` replaced by ``_``. The default cases are the three that ``racecheck_tuning_table``
measures (``numsim-core/examples/racecheck_tuning_table.rs``, which runs this
script itself for missing fixtures). For each case this runs the first racecheck phase through the public v2
path up to the native call. It writes ``OUT_DIR/<case>.{module,inputs,kw}.json``,
which is exactly what the native runner receives, and then stops before execution.
``<case>.key`` holds the sha256 of the module JSON plus the inputs JSON. When the
key is unchanged the files are not rewritten (the module itself comes from the
module cache), so a rerun after an unrelated change is cheap. The fixtures are
build artefacts: ``tuning_table`` writes them under ``core-rs/target/`` and they
are never committed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# examples -> numsim-core -> core-rs -> numsim -> tirx_harness (package) -> src -> tirx_harness -> repo
REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO / "tirx_harness"))

from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES  # noqa: E402
from tirx_harness.numsim import v2  # noqa: E402
from tirx_harness.numsim.v2 import run as v2run  # noqa: E402

DEFAULT_CASES = ("fp16_bf16_gemm", "radix_topk_multi_cta", "gdn_decode_bf16_wide_vec_mtp")


class _Recorded(Exception):
    pass


def _hex(x) -> str | None:
    return bytes(x).hex() if x is not None else None


def _encode_inputs(inputs) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, tuple):
            tag = v[0]
            if tag == "buffer":
                out[k] = {"kind": "buffer", "hex": _hex(v[1]), "mask": _hex(v[2]) if len(v) > 2 else None}
            elif tag == "scalar":
                out[k] = {"kind": "scalar", "v": int(v[1])}
            elif tag == "tensor_map":
                out[k] = {"kind": "tensor_map", "hex": _hex(v[1])}
            elif tag == "tensor_map_of":
                out[k] = {"kind": "tensor_map_of", "base": v[1], "offset": int(v[2]), "hex": _hex(v[3])}
            elif tag == "view":
                out[k] = {"kind": "view", "target": v[1], "offset": int(v[2]), "len": int(v[3])}
            elif tag == "pointer":
                out[k] = {"kind": "pointer", "target": v[1], "offset": int(v[2])}
            else:
                raise SystemExit(f"{k}: input kind {tag!r} is not supported by the fixture format")
        elif isinstance(v, int):
            out[k] = {"kind": "scalar", "v": v}
        else:
            out[k] = {"kind": "buffer", "hex": _hex(v), "mask": None}
    return out


def _mega_moe_case(label: str):
    import os

    from tests.numsim.corpus.kernels.deepgemm import prepare_mega_moe_case
    from tests.numsim.support._tirx_kernels import load_tirx_kernel

    os.environ["TIRX_DEEPGEMM_NUM_SMS_OVERRIDE"] = "148"
    configs = {c["label"]: c for c in load_tirx_kernel("sm100_fp8_fp4_mega_moe").CONFIGS}
    configs["t64_h2048_i1536_e96_k4_g1"] = {
        **configs["t64_h4096_i1536_e96_k4_g1"],
        "hidden": 2048,
        "label": "t64_h2048_i1536_e96_k4_g1",
    }
    engine = v2.Engine(max_workers=16, native_loop_iteration_budget=10_000_000)
    return prepare_mega_moe_case(configs[label]), engine


def record(name: str, out_dir: Path) -> str:
    if name.startswith("mega_moe:"):
        case, engine = _mega_moe_case(name.split(":", 1)[1])
    else:
        entry = next(e for e in CANONICAL_KERNEL_CASES if e.name == name)
        case = entry.prepare()
        engine = v2.Engine(max_workers=entry.engine_max_workers)
    stem = name.replace(":", "_")
    real = v2run.native
    status = "unchanged"

    class Capture:
        def run(self, handle, inputs, **kw):
            nonlocal status
            module = handle.to_json()
            enc = json.dumps(_encode_inputs(inputs), sort_keys=True)
            key = hashlib.sha256(module.encode() + enc.encode()).hexdigest()
            key_file = out_dir / f"{stem}.key"
            if not key_file.exists() or key_file.read_text() != key:
                (out_dir / f"{stem}.module.json").write_text(module)
                (out_dir / f"{stem}.inputs.json").write_text(enc)
                plain = (int, str, float, bool, type(None), list, dict)
                kws = {k: v for k, v in kw.items() if isinstance(v, plain) and k != "codegen_cache_dir"}
                (out_dir / f"{stem}.kw.json").write_text(json.dumps(kws, sort_keys=True))
                key_file.write_text(key)
                status = "recorded"
            raise _Recorded

    v2run.native = lambda: Capture()
    try:
        module = v2.transpile(case.kernel, _analysis_capable=True, _analysis_checker="racecheck")
        engine.run_racecheck_phase(module, case.args, phase_index=0, subset=case.subset, advance_prefix=True)
    except _Recorded:
        return status
    finally:
        v2run.native = real
    raise SystemExit(f"{name}: the racecheck phase never reached the native runner")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[2:] or DEFAULT_CASES:
        print(f"{name}: {record(name, out_dir)}")


if __name__ == "__main__":
    main()
