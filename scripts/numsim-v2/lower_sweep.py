"""Lower every captured PrimFunc with the v2 lowering and rank the residuals.

Usage (from ``tirx_harness/``)::

    $PY ../scripts/numsim-v2/lower_sweep.py CAPTURE_DIR [--filter REGEX] [--strict]
        [--modules OUT_DIR] [--top N]

``--filter`` selects kernels whose capturing test node id matches (the corpus
subset is ``corpus|wiki|microtests|canonical_cases``). ``--modules`` writes one
``Module`` JSON per kernel for ``validate.sh``. Exit status 1 if any kernel
raised an exception (a lowering bug, distinct from an unsupported construct).
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
import traceback
import warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--filter", default=".")
    parser.add_argument("--modules")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    import tvm
    from tirx_harness.numsim.v2.lowering import lower_module

    index: dict[str, set[str]] = {}
    for name in glob.glob(os.path.join(args.capture, "index.*.jsonl")):
        for line in open(name):
            row = json.loads(line)
            index.setdefault(row["hash"], set()).add(row["nodeid"] or "")
    pattern = re.compile(args.filter)
    paths = sorted(
        p for p in glob.glob(os.path.join(args.capture, "*.json"))
        if any(pattern.search(n) for n in index.get(os.path.basename(p)[:-5], ()))
    )
    if args.modules:
        os.makedirs(args.modules, exist_ok=True)
    clean = errors = 0
    reasons: collections.Counter[str] = collections.Counter()
    for path in paths:
        digest = os.path.basename(path)[:-5]
        try:
            func = tvm.ir.load_json(open(path).read())
        except Exception:
            continue
        try:
            module = lower_module(func, strict=False)
        except Exception:
            errors += 1
            print(f"ERROR {digest}", file=sys.stderr)
            traceback.print_exc(limit=-3)
            continue
        program = module.kernels[0]
        if not program.unsupported:
            clean += 1
        for reason in {re.sub(r"site#\d+ ", "", r) for r in program.unsupported}:
            reasons[re.sub(r"(unknown buffer|variable|buffer) \S+", r"\1 X", reason)[:150]] += 1
        if args.modules:
            with open(os.path.join(args.modules, digest + ".json"), "w") as handle:
                handle.write(module.to_json())
    print(f"{len(paths)} kernels; {clean} lower with no Unsupported; {errors} lowering exceptions")
    for reason, count in reasons.most_common(args.top):
        print(f"{count:5d}  {reason}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
