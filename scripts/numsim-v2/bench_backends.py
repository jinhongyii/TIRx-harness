"""Performance comparison: legacy vs the v2 interpreter.

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    $PY ../scripts/numsim-v2/bench_backends.py run   [--cases REGEX] [--modes numsim,racecheck,synccheck]
        [--workers 1,8,32] [--repeats 3] [--max-load 40] [--v2-package DIR]
        [--results DIR] [--case-timeout S] [--force]
    $PY ../scripts/numsim-v2/bench_backends.py render [--results DIR] [--json OUT] [--md OUT]
        [--criterion FILE ...] [--legacy-perf FILE]

``run`` measures every canonical case (``tests/numsim/corpus/canonical_cases.py``)
in each mode whose v2 result matches the frozen legacy conformance snapshot
(``tests/conformance/snapshots``), for the variants

    legacy | v2-interp   x   max_workers in --workers

The v2 codegen backend was measured with this script (plan 2.3), lost, and
was deleted; docs/development/backend-comparison.md keeps that measurement.

Each case runs in its own child process (an abort or an OOM loses one
case, not the sweep) and writes ``<results>/<case>.json``; rows already present
are skipped unless ``--force``. Before every case the host load is checked: the
run waits while the 1-minute load average exceeds ``--max-load`` and records the
CPU count, load average and instantaneous idle (tests/CLAUDE.md preflight).

Per run the timed phases are

* v2: ``transpile`` (module cache hit), ``bind`` (``canonicalize_inputs``),
  ``run`` / ``check`` from ``numsim_core_py.run(...)["timing"]``, ``report`` (the rest of the Python wall: output decoding,
  payload rendering), and ``engine`` = the whole ``Engine.run`` /
  phase-loop wall clock (bind + native + report). Cold lowering
  (``NUMSIM_V2_NO_CACHE=1``) is measured once per case.
* legacy: ``transpile`` (artifact cache hit) and ``engine`` = ``Engine.run`` /
  phase-loop wall clock (legacy has no per-phase timing).

Each configuration repeats ``--repeats`` times interleaved across variants and
keeps the minimum. Every run's normalized result (conformance snapshot
projection) must equal the legacy snapshot (v2: after
``snapshot.relax_unanchored``) and all v2 runs must be identical to each other;
otherwise the row is aborted and the reason recorded.

``render`` merges the per-case files into ``docs/development/backend-comparison.json``
and the markdown report.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "tirx_harness"
MODES = ("numsim", "racecheck", "synccheck")
VARIANTS = ("legacy", "interp")
LONG_RUN_S = 60.0  # one repetition only for configurations slower than this

# Instr families (numsim-core/src/program.rs `enum Instr`).
FAMILIES = {
    "control": "Nop If Else EndIf LoopBegin LoopIf LoopEnd Break Continue Exit Assert Unsupported",
    "scalar": "Mov ReadSpecial ReadParam Unary Binary Ternary Compare Select Cast Ptx LoadRegIndexed StoreRegIndexed",
    "warp": "Shfl Vote Redux Elect WarpSync",
    "memory": "LdMatrix StMatrix Load Store LoadAddr StoreAddr AddrOf Atom StBulk Discard Cvta Isspacep Mapa GetCtaRank",
    "async_tma": "CpAsync AsyncCommit AsyncWait CpAsyncMbarArrive BulkCopy Tma StAsync TensorMapReplace TensorMapCopyFence",
    "sync": "Barrier ClusterArrive ClusterWait GridSync MbarInit MbarInval MbarArrive MbarTx MbarTestWait MbarWait "
    "MbarQuery Fence SetMaxNReg WaitUntil GridDepControl ClcTryCancel",
    "tcgen_mma": "TcgenAlloc TcgenDealloc TcgenRelinquish TcgenCommit TcgenLd TcgenSt TcgenWait TcgenCp TcgenMma",
    "tile": "Tile",
}
FAMILY_OF = {name: family for family, names in FAMILIES.items() for name in names.split()}
# "Heavy" instructions: one dispatch does a whole tile / bulk transfer / MMA.
HEAVY = set(FAMILIES["async_tma"].split()) | set(FAMILIES["tcgen_mma"].split()) | {"Tile", "LdMatrix", "StMatrix", "StBulk"}


# --------------------------------------------------------------------------
# Host preflight


def _cpu_times() -> tuple[int, int]:
    fields = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = fields[3] + fields[4]
    return idle, sum(fields)


def preflight(sample_s: float = 1.0) -> dict[str, Any]:
    idle0, total0 = _cpu_times()
    time.sleep(sample_s)
    idle1, total1 = _cpu_times()
    load = os.getloadavg()
    return {
        "nproc": os.cpu_count(),
        "loadavg": [round(x, 2) for x in load],
        "idle_pct": round(100.0 * (idle1 - idle0) / max(1, total1 - total0), 1),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def wait_for_quiet(max_load: float, log=print) -> dict[str, Any]:
    while True:
        state = preflight()
        if state["loadavg"][0] <= max_load:
            return state
        log(f"  host busy (load {state['loadavg']}, idle {state['idle_pct']}%); waiting 30s")
        time.sleep(30)


# --------------------------------------------------------------------------
# Imports


def import_v2(package_dir: str | None):
    """The v2 package: the in-tree one, or a frozen copy (``--v2-package``)."""

    if not package_dir:
        return importlib.import_module("tirx_harness.numsim.v2")
    path = Path(package_dir).resolve()
    sys.path.insert(0, str(path.parent))
    return importlib.import_module(path.name)


def instr_mix(module: Any) -> dict[str, Any]:
    """Static ``Instr`` counts per family over every kernel of a v2 module."""

    document = json.loads(module.handle.to_json())
    counts: dict[str, int] = {}
    variants: dict[str, int] = {}
    for kernel in document["kernels"]:
        for instr in kernel["code"]:
            name = instr if isinstance(instr, str) else next(iter(instr))
            variants[name] = variants.get(name, 0) + 1
            family = FAMILY_OF.get(name, "other")
            counts[family] = counts.get(family, 0) + 1
    total = sum(counts.values())
    heavy = sum(n for name, n in variants.items() if name in HEAVY)
    return {
        "kernels": len(document["kernels"]),
        "instrs": total,
        "families": dict(sorted(counts.items())),
        "heavy": heavy,
        "heavy_pct": round(100.0 * heavy / max(1, total), 2),
        "variants": dict(sorted(variants.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------------
# One case (child process)


class Timer:
    def __init__(self):
        self.t = 0.0

    @contextlib.contextmanager
    def __call__(self):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.t += time.perf_counter() - started


def _synccheck_kwargs(numsim: Any, entry: Any) -> dict[str, Any]:
    from tests.conformance import snapshot as snap

    return {
        "coverage_bounds": numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0),
        "resource_limits": snap._synccheck_limits(numsim, entry.synccheck_max_diagnostic_bytes),
    }


def _phases(numsim: Any, engine: Any, module: Any, case: Any, entry: Any, mode: str) -> list[dict[str, Any]]:
    out = []
    for phase_index in range(len(module.spec.kernels)):
        if mode == "racecheck":
            result = engine.run_racecheck_phase(
                module, case.args, phase_index=phase_index, subset=case.subset, advance_prefix=True
            )
        else:
            result = engine.run_synccheck_phase(
                module, case.args, phase_index=phase_index, subset=case.subset, advance_prefix=True,
                **_synccheck_kwargs(numsim, entry),
            )
        out.append(result.to_dict())
    return out


def _legacy_transpile(numsim: Any, case: Any, mode: str) -> Any:
    if mode == "numsim":
        return numsim.transpile(case.kernel)
    if mode == "racecheck":
        return numsim.transpile(case.kernel, _analysis_capable=True, _analysis_checker="racecheck")
    return numsim.transpile(case.kernel, _analysis_capable=True, _default_generated_opt_level=3)


def run_legacy(entry: Any, mode: str, workers: int, expected: Any) -> tuple[dict, dict]:
    from dataclasses import replace

    from tirx_harness import numsim
    from tests.conformance import snapshot as snap

    case = entry.prepare()
    t_tr = Timer()
    with t_tr():
        module = _legacy_transpile(numsim, case, mode)
    engine = numsim.Engine(max_workers=workers)
    resolver = snap.SourceResolver(module)
    t_eng = Timer()
    if mode == "numsim":
        with t_eng():
            execution = engine._prepare_execution(module, case.args, outputs=case.outputs, assumptions=case.assumptions)
            execution = replace(execution, bindings=execution.bindings.freeze())
            result = engine._execute_prepared(module, execution, subset=case.subset)
        execution.bindings.restore_host_buffers()
        ok = bool(numsim.compare(result, expected, tolerances=case.comparisons).ok)
        norm = snap.normalize_numsim(result.outputs, result.diagnostics, reference_ok=ok, resolver=resolver)
    else:
        with t_eng():
            payloads = _phases(numsim, engine, module, case, entry, mode)
        norm = {"phases": [snap.normalize_analysis_phase(p, resolver) for p in payloads]}
    return norm, {"transpile": t_tr.t, "engine": t_eng.t}


def run_v2(v2: Any, entry: Any, mode: str, workers: int, expected: Any) -> tuple[dict, dict]:
    from tests.conformance import snapshot as snap

    run_mod = sys.modules[v2.Engine.__module__]
    case = entry.prepare()
    t_tr = Timer()
    with t_tr():
        module = v2.transpile(case.kernel)
    engine = v2.Engine(max_workers=workers)
    t_bind, t_native = Timer(), Timer()
    timing = {"run": 0.0, "check": 0.0}
    stats: dict[str, Any] = {}

    def on_raw(raw: dict) -> None:
        for key in timing:
            timing[key] += float((raw.get("timing") or {}).get(key, 0.0)) / 1e3
        stats.update({k: v for k, v in (raw.get("stats") or {}).items() if isinstance(v, (int, float))})

    original_bind = run_mod.canonicalize_inputs

    def timed_bind(*a, **k):
        with t_bind():
            return original_bind(*a, **k)

    run_mod.canonicalize_inputs = timed_bind
    restore_native = patch_native(run_mod, t_native, on_raw)
    resolver = snap.SourceResolver(module)
    t_eng = Timer()
    try:
        if mode == "numsim":
            with t_eng():
                result = engine.run(module, case.args, subset=case.subset, assumptions=case.assumptions,
                                    outputs=case.outputs)
            ok = bool(v2.compare(result, expected, tolerances=case.comparisons).ok)
            norm = snap.normalize_numsim(result.outputs, result.diagnostics, reference_ok=ok, resolver=resolver)
            engine_timing = dict(getattr(result, "timing", None) or {})
        else:
            with t_eng():
                payloads = _phases(v2, engine, module, case, entry, mode)
            norm = {"phases": [snap.normalize_analysis_phase(p, resolver) for p in payloads]}
            engine_timing = dict(payloads[0].get("timing") or {}) if payloads else {}
    finally:
        run_mod.canonicalize_inputs = original_bind
        restore_native()
    times = {
        "transpile": t_tr.t,
        "engine": t_eng.t,
        "bind": t_bind.t,
        "native": t_native.t,
        "run": timing["run"],
        "check": timing["check"],
        "report": max(0.0, t_eng.t - t_bind.t - t_native.t),
    }
    return norm, {"times": times, "stats": stats, "engine_timing_ms": engine_timing}


def patch_native(run_mod: Any, timer: Timer, on_raw) -> Any:
    """Route ``numsim_core_py.run`` through a proxy that times it and reports
    its raw result; returns the undo."""

    original = run_mod.native
    real = original()

    class Proxy:
        def __getattr__(self, name):
            return getattr(real, name)

        def run(self, *a, **k):
            with timer():
                raw = real.run(*a, **k)
            on_raw(raw)
            return raw

    proxy = Proxy()
    run_mod.native = lambda: proxy

    def restore():
        run_mod.native = original

    return restore


def bench_case(args: argparse.Namespace) -> dict[str, Any]:
    os.chdir(HARNESS)
    sys.path.insert(0, str(HARNESS))
    v2 = import_v2(args.v2_package)
    from tests.conformance import snapshot as snap
    from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES

    entry = next(e for e in CANONICAL_KERNEL_CASES if e.name == args.case)
    out: dict[str, Any] = {"case": entry.name, "modes": {}, "preflight": [wait_for_quiet(args.max_load)]}

    # Static facts and one-off costs (module shared by every v2 mode).
    case = entry.prepare()
    t0 = time.perf_counter()
    previous = os.environ.get("NUMSIM_V2_NO_CACHE")
    os.environ["NUMSIM_V2_NO_CACHE"] = "1"
    try:
        module = v2.transpile(case.kernel)
        out["lower_cold_s"] = time.perf_counter() - t0
    except Exception as error:  # noqa: BLE001
        out["error"] = f"v2 transpile: {type(error).__name__}: {error}"[:2000]
        return out
    finally:
        if previous is None:
            os.environ.pop("NUMSIM_V2_NO_CACHE", None)
        else:
            os.environ["NUMSIM_V2_NO_CACHE"] = previous
    v2.transpile(case.kernel)  # populate the module cache
    out["instr_mix"] = instr_mix(module)
    del case, module

    workers = [int(w) for w in args.workers.split(",")]
    for mode in args.modes.split(","):
        stored = snap.load_snapshot(entry.name, mode)
        row: dict[str, Any] = {"status": "pending", "runs": {}}
        out["modes"][mode] = row
        if stored is None or "error" in stored:
            row["status"] = "skipped: no legacy oracle"
            continue
        stored = {k: v for k, v in stored.items() if k not in ("schema", "case", "mode")}
        # v2's oracle: ``<mode>.delta.json`` (the legacy snapshot corrected by a
        # behaviour-delta row) when present, else the legacy snapshot.
        delta_path = snap.delta_snapshot_path(entry.name, mode) if hasattr(snap, "delta_snapshot_path") else None
        v2_oracle = stored
        if delta_path is not None and delta_path.exists():
            delta = json.loads(delta_path.read_text())
            row["oracle"] = f"delta {delta.get('delta')}"
            v2_oracle = {k: v for k, v in delta.items() if k not in ("schema", "case", "mode", "delta")}
        row["preflight"] = wait_for_quiet(args.max_load)
        try:
            expected = copy.deepcopy(entry.prepare().reference()) if mode == "numsim" else None
        except Exception as error:  # noqa: BLE001
            row["status"] = f"skipped: reference failed: {type(error).__name__}"
            continue
        reference_norm: dict | None = None

        def check(variant: str, norm: dict) -> str | None:
            nonlocal reference_norm
            if variant == "legacy":
                return None if norm == stored else (
                    "legacy result differs from its own snapshot:\n" + snap.diff_snapshots(stored, norm)[:4000])
            relaxed = snap.relax_unanchored(v2_oracle, norm)
            if relaxed != v2_oracle:
                return (f"{variant} does not match its oracle ({row.get('oracle', 'legacy snapshot')}):\n"
                        + snap.diff_snapshots(v2_oracle, relaxed)[:4000])
            if reference_norm is None:
                reference_norm = norm
            elif norm != reference_norm:
                return f"{variant} differs from the first v2 result:\n" + snap.diff_snapshots(reference_norm, norm)[:4000]
            return None

        def one(variant: str, w: int) -> dict:
            if variant == "legacy":
                norm, rec = run_legacy(entry, mode, w, expected)
                rec = {"times": rec}
            else:
                norm, rec = run_v2(v2, entry, mode, w, expected)
            problem = check(variant, norm)
            if problem:
                raise RowAbort(problem, norm)
            rec["load1"] = round(os.getloadavg()[0], 2)
            return rec

        try:
            # Eligibility: v2-interp must match the legacy snapshot.
            first = one("interp", workers[0])
            row["eligibility_run"] = first
            for w in workers:
                samples: dict[str, list[dict]] = {v: [] for v in args.variants}
                reps = args.repeats
                rep = 0
                while rep < reps:
                    for variant in args.variants:
                        if rep > 0 and len(samples[variant]) == 1 and samples[variant][0]["times"]["engine"] > LONG_RUN_S:
                            continue
                        samples[variant].append(one(variant, w))
                    rep += 1
                for variant, recs in samples.items():
                    best = min(recs, key=lambda r: r["times"]["engine"])
                    row["runs"].setdefault(variant, {})[str(w)] = {
                        "min": best["times"],
                        "samples_engine": [round(r["times"]["engine"], 6) for r in recs],
                        "stats": best.get("stats", {}),
                        "load1": [r["load1"] for r in recs],
                    }
            row["status"] = "ok"
        except RowAbort as abort:
            row["status"] = f"aborted: {abort.reason}"
        except NotImplementedError as error:
            row["status"] = f"skipped: not implemented: {str(error)[:300]}"
        except Exception as error:  # noqa: BLE001
            row["status"] = f"aborted: {type(error).__name__}: {str(error)[:600]}"
        print(f"  {entry.name}/{mode}: {row['status']}", flush=True)
    out["preflight"].append(preflight())
    return out


class RowAbort(Exception):
    def __init__(self, reason: str, norm: Any):
        super().__init__(reason)
        self.reason = reason
        self.norm = norm



def selected_cases(args: argparse.Namespace) -> list[str]:
    sys.path.insert(0, str(HARNESS))
    os.chdir(HARNESS)
    from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES

    pattern = re.compile(args.cases)
    names = [e.name for e in CANONICAL_KERNEL_CASES if pattern.search(e.name)]
    if args.status_md:
        wanted = set(args.modes.split(","))
        status = parse_status(Path(args.status_md))
        names = [n for n in names if any(status.get(n, {}).get(m) == "match" for m in wanted)]
    return names


# --------------------------------------------------------------------------
# Driver


def cmd_run(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(HARNESS))
    os.chdir(HARNESS)
    from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.cases)
    names = [e.name for e in CANONICAL_KERNEL_CASES if pattern.search(e.name)]
    if args.status_md:
        # Only cases whose v2 status is "match" in at least one requested mode.
        wanted = set(args.modes.split(","))
        status = parse_status(Path(args.status_md))
        names = [n for n in names if any(status.get(n, {}).get(m) == "match" for m in wanted)]
    print(f"{len(names)} cases; preflight {preflight()}", flush=True)
    for name in names:
        path = results / f"{name}.json"
        if path.exists() and not args.force:
            continue
        modes = args.modes
        if args.status_md:
            modes = ",".join(m for m in args.modes.split(",") if status.get(name, {}).get(m) == "match")
        cmd = [sys.executable, __file__, "case", name, "--modes", modes, "--workers", args.workers,
               "--repeats", str(args.repeats), "--max-load", str(args.max_load),
               "--variants", ",".join(args.variants)]
        if args.v2_package:
            cmd += ["--v2-package", args.v2_package]
        print(f"[{time.strftime('%H:%M:%S')}] {name} ({modes})", flush=True)
        started = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.case_timeout)
            stdout, code = proc.stdout, proc.returncode
            stderr = proc.stderr
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr, code = f"timeout after {args.case_timeout}s", -1
        marker = "@@RESULT@@"
        if marker in stdout:
            data = json.loads(stdout.split(marker, 1)[1])
        else:
            data = {"case": name, "error": f"child exited {code}: {stderr[-3000:]}", "modes": {}}
        data["wall_s"] = time.perf_counter() - started
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
        for line in stdout.split(marker, 1)[0].splitlines():
            if line.startswith("  "):
                print(line, flush=True)
        if "error" in data:
            print(f"  error: {data['error'][:300]}", flush=True)
    return 0


def cmd_case(args: argparse.Namespace) -> int:
    data = bench_case(args)
    sys.stdout.write("@@RESULT@@" + json.dumps(data, default=str))
    sys.stdout.flush()
    return 0


def parse_status(path: Path) -> dict[str, dict[str, str]]:
    status: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\| `([^`]+)` \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            status[m.group(1)] = {
                mode: cell.strip().split(" ")[0] if cell.strip() != "no oracle" else "no oracle"
                for mode, cell in zip(MODES, m.groups()[1:])
            }
    return status


# --------------------------------------------------------------------------
# Rendering


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def fmt_s(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1e-3:
        return f"{value * 1e6:.0f}us"
    if value < 1:
        return f"{value * 1e3:.1f}ms"
    return f"{value:.2f}s"


def fmt_x(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _engine(row: dict, variant: str, w: str) -> float | None:
    run = row.get("runs", {}).get(variant, {}).get(w)
    return run["min"]["engine"] if run else None


def _core(row: dict, variant: str, w: str) -> float | None:
    """v2 native run + check (no build, no Python)."""

    run = row.get("runs", {}).get(variant, {}).get(w)
    if not run or "run" not in run["min"]:
        return None
    return run["min"]["run"] + run["min"]["check"]


def _pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""

    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))]


def _loaded(row: dict, limit: float = 40.0) -> bool:
    """Any sample of the row taken while the 1-minute load average exceeded ``limit``."""

    return any(x > limit for runs in row.get("runs", {}).values() for run in runs.values()
               for x in run.get("load1", ()))


def _ns_per_instr(row: dict, w: str) -> float | None:
    """Interp native run time per executed warp instruction (``stats.instrs``)."""

    run = row.get("runs", {}).get("interp", {}).get(w)
    if not run or not run.get("stats", {}).get("instrs"):
        return None
    return 1e9 * run["min"]["run"] / run["stats"]["instrs"]


def cmd_render(args: argparse.Namespace) -> int:
    results = Path(args.results)
    cases = [json.loads(p.read_text()) for p in sorted(results.glob("*.json"))]
    meta = json.loads(Path(args.meta).read_text()) if args.meta and Path(args.meta).exists() else {}
    criterion = {Path(f).stem: Path(f).read_text() for f in args.criterion or ()}
    legacy_perf = json.loads(Path(args.legacy_perf).read_text()) if args.legacy_perf else None
    workers = sorted({w for c in cases for row in c.get("modes", {}).values()
                      for runs in row.get("runs", {}).values() for w in runs}, key=int)
    present = [v for v in VARIANTS if any(v in row.get("runs", {}) for c in cases
                                          for row in c.get("modes", {}).values())]
    doc = {"meta": meta, "cases": cases, "legacy_perf_tests": legacy_perf}
    Path(args.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    md: list[str] = []
    add = md.append
    add("---\norphan: true\n---\n")
    add("# NumSim backend comparison (" + " vs ".join(present) + ")\n")
    add("Generated by `scripts/numsim-v2/bench_backends.py`; raw data in `backend-comparison.json`. "
        "Plan section 2.3: two backends behind one switch, compare on the corpus per mode, delete the loser.\n")
    if meta:
        add("## Setup\n")
        for key, value in meta.items():
            add(f"- **{key}**: {value}")
        add("")
    add("Times are the minimum over the repetitions of the engine wall clock: legacy `Engine.run` / the "
        "phase loop (artifact already compiled); v2 the same public calls (input binding + native run + "
        "report rendering). `core` columns are the v2 native `run + check` only (from `numsim_core_py.run()['timing']`). "
        "Speedup = legacy / variant (> 1 means faster than legacy).\n")

    if args.recommendation and Path(args.recommendation).exists():
        add(Path(args.recommendation).read_text())
    # Summary.
    add("## Summary\n")
    add("Geometric means over rows measured in every variant (speedup vs legacy, engine wall clock).\n")
    others = [v for v in present if v != "legacy"]
    head = ["mode", "workers", "rows", *present, *(f"{v} vs legacy" for v in others)]
    add("| " + " | ".join(head) + " |")
    add("|" + " --- |" * len(head))
    summary: dict[str, Any] = {}
    for mode in MODES:
        for w in workers:
            rows = [c["modes"][mode] for c in cases if c.get("modes", {}).get(mode, {}).get("status") == "ok"]
            rows = [r for r in rows if all(_engine(r, v, w) for v in present)]
            if not rows:
                continue
            g = {v: geomean([_engine(r, v, w) for r in rows]) for v in present}
            sp = {v: geomean([_engine(r, "legacy", w) / _engine(r, v, w) for r in rows]) for v in others}
            cells = [mode, w, str(len(rows)), *(fmt_s(g[v]) for v in present), *(fmt_x(sp[v]) for v in others)]
            summary[f"{mode}/{w}"] = {"rows": len(rows), "geomean_s": g, "speedup_vs_legacy": sp}
            add("| " + " | ".join(cells) + " |")
    add("")

    # Instruction mix correlation.
    _render_mix(add, cases, workers)

    # Per-case tables.
    _render_cases(add, cases, workers)

    _render_tail(add, cases, legacy_perf, criterion)
    Path(args.md).write_text("\n".join(md) + "\n")
    print(f"wrote {args.json} and {args.md}")
    return 0


def _render_mix(add, cases, workers) -> None:
    add("## Instruction mix\n")
    add("Static `Instr` counts of the lowered module (`numsim_core_py.Module.to_json`), per family. "
        "`heavy` = tile / TMA / bulk copy / tcgen05 / ldmatrix instructions whose single dispatch does a "
        "whole tile of work; the rest are per-lane scalar, control, warp and memory instructions. Rows bucketed by heavy share; geomeans at the largest "
        "worker count measured:\n")
    w = workers[-1] if workers else None
    add("| mode | bucket | rows | interp ns per warp instr (median) | interp vs legacy (engine) |")
    add("| --- | --- | --- | --- | --- |")
    buckets = (("pure scalar (no heavy instrs)", 0, 1e-9), ("scalar-dominated (heavy < 1.5%)", 1e-9, 1.5),
               ("tile/TMA/MMA-rich (heavy >= 1.5%)", 1.5, 101))
    for mode in MODES:
        for label, lo, hi in buckets:
            rows = [c["modes"][mode] for c in cases if c.get("modes", {}).get(mode, {}).get("status") == "ok"
                    and lo <= c.get("instr_mix", {}).get("heavy_pct", 0) < hi]
            rows = [r for r in rows if _core(r, "interp", w)]
            if not rows:
                continue
            il = geomean([_engine(r, "legacy", w) / _engine(r, "interp", w) for r in rows if _engine(r, "legacy", w)])
            nspi = [_ns_per_instr(r, w) for r in rows]
            nspi = [x for x in nspi if x]
            add(f"| {mode} | {label} | {len(rows)} | {statistics.median(nspi):.0f} | {fmt_x(il)} |"
                if nspi else f"| {mode} | {label} | {len(rows)} | - | {fmt_x(il)} |")
    add("")



def _render_cases(add, cases, workers) -> None:
    for mode in MODES:
        for w in workers:
            rows = [(c, c["modes"][mode]) for c in cases if mode in c.get("modes", {})
                    and c["modes"][mode].get("status") == "ok" and str(w) in c["modes"][mode].get("runs", {}).get("interp", {})]
            if not rows:
                continue
            add(f"## {mode}, max_workers={w}\n")
            add("`(L)` = measured under load: some repetition ran while the 1-minute load average exceeded 40.\n")
            head = ["case", "heavy%", "dyn instrs", "ns/instr", "legacy", "interp", "interp/legacy", "core interp"]
            add("| " + " | ".join(head) + " |")
            add("|" + " --- |" * len(head))
            for c, r in sorted(rows, key=lambda cr: cr[0]["case"]):
                leg, it = (_engine(r, v, str(w)) for v in VARIANTS)
                instrs = r["runs"]["interp"][str(w)].get("stats", {}).get("instrs")
                ci = _core(r, "interp", str(w))
                nspi = _ns_per_instr(r, str(w))
                cells = [f"`{c['case']}`{' (L)' if _loaded(r) else ''}", f"{c.get('instr_mix', {}).get('heavy_pct', 0):.1f}",
                         str(instrs) if instrs is not None else "-", f"{nspi:.0f}" if nspi else "-", fmt_s(leg), fmt_s(it)]
                cells.append(fmt_x(leg / it if leg and it else None))
                cells.append(fmt_s(ci))
                add("| " + " | ".join(cells) + " |")
            add("")


def _render_tail(add, cases, legacy_perf, criterion) -> None:
    # Rows not measured.
    add("## Rows not measured\n")
    add("| case | mode | status |")
    add("| --- | --- | --- |")
    for c in cases:
        if c.get("error"):
            add(f"| `{c['case']}` | all | {c['error'][:200].replace('|', '/').splitlines()[0]} |")
        for mode, row in c.get("modes", {}).items():
            if row.get("status") != "ok":
                add(f"| `{c['case']}` | {mode} | {row.get('status', '')[:200].replace('|', '/')} |")
    add("")

    if legacy_perf:
        add("## Legacy performance tests (Mega-MoE)\n")
        add(legacy_perf.get("markdown", ""))
        add("")
    for name, text in criterion.items():
        add(f"## Criterion: `{name}`\n")
        add("```text\n" + text.strip() + "\n```\n")


# --------------------------------------------------------------------------
# Mega-MoE perf-budget workloads (not in the corpus sweep: too large)

MEGA_CONFIGS = {
    # alias: (registry label, derived-from label or None, overrides)
    "small": ("p1_tok2_h1024_i512_e2_k1_bm16", None, {}),
    "twenty_four_experts": ("t8_h1024_i512_e24_k2_g1", None, {}),
    "medium": ("t64_h2048_i1536_e96_k4_g1", "t64_h4096_i1536_e96_k4_g1", {"hidden": 2048}),
    "large": ("t64_h4096_i1536_e96_k4_g1", None, {}),
}
MEGA_NUM_SMS = 148  # as the legacy racecheck perf test


def mega_one(args: argparse.Namespace) -> dict[str, Any]:
    """Child: one (config, mode, impl, workers) Mega-MoE run, timed like the
    legacy perf tests (transpile outside the timed region)."""

    os.chdir(HARNESS)
    sys.path.insert(0, str(HARNESS))
    os.environ["TIRX_DEEPGEMM_NUM_SMS_OVERRIDE"] = str(MEGA_NUM_SMS)
    os.environ.pop("NUMSIM_PROFILE", None)
    from tests.numsim.corpus.kernels.deepgemm import prepare_mega_moe_case
    from tests.numsim.support._tirx_kernels import load_tirx_kernel

    label, base, overrides = MEGA_CONFIGS[args.config]
    configs = {c["label"]: c for c in load_tirx_kernel("sm100_fp8_fp4_mega_moe").CONFIGS}
    config = {**configs[base or label], **overrides, "label": label}
    if args.impl == "legacy":
        from tirx_harness import numsim as impl
    else:
        impl = import_v2(args.v2_package)
    case = prepare_mega_moe_case(config)
    out: dict[str, Any] = {"config": args.config, "label": label, "mode": args.mode, "impl": args.impl,
                           "workers": args.workers_one, "preflight": preflight(0.5)}
    cache = Path(os.environ.get("NUMSIM_CACHE_DIR", Path.home() / ".cache/tirx-harness/numsim")) / "bench-mega-moe"
    engine = impl.Engine(max_workers=args.workers_one, native_loop_iteration_budget=10_000_000)
    if args.mode == "numsim":
        module = impl.transpile(case.kernel)
        expected = case.reference()
        started = time.perf_counter()
        result = engine.run(module, case.args, subset=case.subset, assumptions=case.assumptions, outputs=case.outputs)
        out["elapsed_s"] = time.perf_counter() - started
        out["reference_ok"] = bool(impl.compare(result, expected, tolerances=case.comparisons).ok)
        out["verdict"] = str(getattr(result, "verdict", None))
        out["diagnostics"] = len(result.diagnostics)
    else:
        module = impl.transpile(case.kernel, cache_dir=cache, _analysis_capable=True, _analysis_checker="racecheck")
        started = time.perf_counter()
        result = engine.run_racecheck_phase(module, case.args, phase_index=0, subset=case.subset)
        out["elapsed_s"] = time.perf_counter() - started
        payload = result.to_dict()
        out["verdict"] = result.verdict
        out["findings"] = len(result.findings)
        out["advisory_kinds"] = sorted(str(a.get("kind")) for a in result.advisories)
        out["incomplete"] = len(payload.get("incomplete") or [])
    out["load1_after"] = round(os.getloadavg()[0], 2)
    return out


def cmd_mega(args: argparse.Namespace) -> int:
    target = Path(args.results) / "mega"
    target.mkdir(parents=True, exist_ok=True)
    for config in args.configs.split(","):
        for mode in args.mega_modes.split(","):
            for w in [int(x) for x in args.mega_workers.split(",")]:
                for impl in args.impls.split(","):
                    path = target / f"{config}.{mode}.{impl}.{w}.json"
                    if path.exists() and not args.force:
                        continue
                    state = wait_for_quiet(args.max_load)
                    cmd = [sys.executable, __file__, "mega-one", config, mode, impl, str(w)]
                    if args.v2_package:
                        cmd += ["--v2-package", args.v2_package]
                    started = time.perf_counter()
                    try:
                        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.run_timeout)
                        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
                    except subprocess.TimeoutExpired:
                        stdout, stderr, code = "", f"timeout after {args.run_timeout:.0f}s", -1
                    if "@@RESULT@@" in stdout:
                        data = json.loads(stdout.split("@@RESULT@@", 1)[1])
                    else:
                        data = {"config": config, "mode": mode, "impl": impl, "workers": w,
                                "error": stderr[-2000:] or f"exit {code}",
                                "timeout": code == -1, "preflight": state}
                    data["wall_s"] = time.perf_counter() - started
                    path.write_text(json.dumps(data, indent=1, sort_keys=True))
                    shown = f"{data['elapsed_s']:.2f}s" if "elapsed_s" in data else (
                        "TIMEOUT" if data.get("timeout") else "ERROR " + data.get("error", "")[-200:])
                    print(f"[{time.strftime('%H:%M:%S')}] mega {config} {mode} {impl} w={w}: {shown} "
                          f"verdict={data.get('verdict')} load={state['loadavg']}", flush=True)
    return 0


def cmd_mega_one(args: argparse.Namespace) -> int:
    try:
        data = mega_one(args)
    except Exception as error:  # noqa: BLE001
        data = {"config": args.config, "mode": args.mode, "impl": args.impl, "workers": args.workers_one,
                "error": f"{type(error).__name__}: {str(error)[:1500]}"}
    sys.stdout.write("@@RESULT@@" + json.dumps(data, default=str))
    sys.stdout.flush()
    return 0


def mega_rows(results: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((results / "mega").glob("*.json"))]


# --------------------------------------------------------------------------
# Regression list (interp slower than legacy)


def owner_guess(case: dict, mode: str, w: str) -> str:
    """Heuristic owner of an interp-slower-than-legacy row.

    * ``scheduler/W2``: at more than one worker, legacy gets >= 1.5x faster
      from 1 worker to the largest count while interp gets < 1.2x faster.
    * ``racecheck/W5`` / ``synccheck/W6``: the checker mode is slower than
      legacy at 1 worker while NumSim of the same case is not.
    * ``numsim/W2`` otherwise.
    """

    row = case["modes"][mode]
    ws = sorted(row["runs"]["interp"], key=int)
    lo, hi = ws[0], ws[-1]
    leg_scale = _engine(row, "legacy", lo) / _engine(row, "legacy", hi)
    int_scale = _engine(row, "interp", lo) / _engine(row, "interp", hi)
    if w != lo and leg_scale >= 1.5 and int_scale < 1.2:
        return "scheduler/W2"
    if mode in ("racecheck", "synccheck") and _engine(row, "interp", lo) > _engine(row, "legacy", lo):
        numsim = case["modes"].get("numsim", {})
        numsim_slow = (numsim.get("status") == "ok" and lo in numsim["runs"].get("interp", {})
                       and _engine(numsim, "interp", lo) > _engine(numsim, "legacy", lo))
        if not numsim_slow:
            return "racecheck/W5" if mode == "racecheck" else "synccheck/W6"
    return "numsim/W2"


def _mega_regressions(results: Path) -> list[tuple]:
    """Mega-MoE rows where interp is slower than legacy (a timeout counts,
    with the cap as a lower bound on interp). Owner: racecheck/W5 when the
    racecheck slowdown is more than 1.5x the NumSim slowdown of the same
    config and workers; scheduler/W2 when legacy scales with workers and
    interp does not; numsim/W2 otherwise."""

    data = mega_rows(results)
    index = {(d["config"], d["mode"], d["impl"], int(d["workers"])): d for d in data}

    def t(config, mode, impl, w):
        d = index.get((config, mode, impl, w))
        if d is None:
            return None
        if "elapsed_s" in d:
            return float(d["elapsed_s"])
        return float(d["wall_s"]) if d.get("timeout") else None

    out = []
    ws = sorted({k[3] for k in index})
    for (config, mode, impl, w), d in sorted(index.items()):
        if impl != "interp":
            continue
        leg, it = t(config, mode, "legacy", w), t(config, mode, "interp", w)
        if not leg or not it or it <= leg:
            continue
        ratio = it / leg
        owner = "numsim/W2"
        if ws and w != ws[0]:
            l1, i1 = t(config, mode, "legacy", ws[0]), t(config, mode, "interp", ws[0])
            if l1 and i1 and l1 / leg >= 1.5 and i1 / it < 1.2 and not d.get("timeout"):
                owner = "scheduler/W2"
        if mode == "racecheck":
            nl, ni = t(config, "numsim", "legacy", w), t(config, "numsim", "interp", w)
            if not (nl and ni) or ratio > 1.5 * (ni / nl):
                owner = "racecheck/W5"
        verdict = "" if d.get("verdict") == index.get((config, mode, "legacy", w), {}).get("verdict") else " (verdict differs)"
        name = f"mega_moe:{config}" + (" (timeout)" if d.get("timeout") else "") + verdict
        out.append((ratio, name, mode, str(w), leg, it, owner, (d.get("preflight") or {}).get("loadavg", [0])[0] > 40))
    return out


def cmd_regressions(args: argparse.Namespace) -> int:
    cases = [json.loads(p.read_text()) for p in sorted(Path(args.results).glob("*.json"))]
    lines = ["case\tmode\tworkers\tlegacy_s\tinterp_s\tratio\towner_guess\tunder_load\tengine"]
    rows = []
    for c in cases:
        for mode, row in c.get("modes", {}).items():
            if row.get("status") != "ok":
                continue
            for w in sorted(row["runs"].get("interp", {}), key=int):
                leg, it = _engine(row, "legacy", w), _engine(row, "interp", w)
                if leg and it and it > leg:
                    rows.append((it / leg, c["case"], mode, w, leg, it, owner_guess(c, mode, w), _loaded(row)))
    rows += _mega_regressions(Path(args.results))
    for ratio, case, mode, w, leg, it, owner, loaded in sorted(rows, key=lambda r: -r[0]):
        shown = f"{it:.4f}" + ("+" if "(timeout)" in case else "")
        lines.append(f"{case}\t{mode}\t{w}\t{leg:.4f}\t{shown}\t{ratio:.2f}\t{owner}\t{'yes' if loaded else 'no'}\t{args.engine}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_root = Path(os.environ.get("NUMSIM_CACHE_DIR", Path.home() / ".cache/tirx-harness/numsim"))

    def common(p):
        p.add_argument("--modes", default=",".join(MODES))
        p.add_argument("--workers", default="1,8,32")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--max-load", type=float, default=40.0)
        p.add_argument("--v2-package", default=None, help="frozen copy of the v2 package (with its .so)")
        p.add_argument("--variants", type=lambda s: s.split(","), default=list(VARIANTS),
                       help=f"comma list of {','.join(VARIANTS)} (default: both)")

    run = sub.add_parser("run")
    common(run)
    run.add_argument("--cases", default=".")
    run.add_argument("--results", default=str(default_root / "bench-backends"))
    run.add_argument("--status-md", default=str(REPO / "docs/development/v2-conformance-status.md"),
                     help="only modes listed as 'match' here ('' = all modes; rows are re-verified anyway)")
    run.add_argument("--case-timeout", type=float, default=3 * 3600)
    run.add_argument("--force", action="store_true")
    case = sub.add_parser("case")
    case.add_argument("case")
    common(case)
    render = sub.add_parser("render")
    render.add_argument("--results", default=str(default_root / "bench-backends"))
    render.add_argument("--json", default=str(REPO / "docs/development/backend-comparison.json"))
    render.add_argument("--md", default=str(REPO / "docs/development/backend-comparison.md"))
    render.add_argument("--meta", default=None, help="JSON object of setup facts for the header")
    render.add_argument("--criterion", nargs="*", help="criterion output text files (one section each)")
    render.add_argument("--legacy-perf", default=None, help="JSON with a 'markdown' key")
    render.add_argument("--recommendation", default=None, help="markdown appended verbatim")
    reg = sub.add_parser("regressions", help="every (case, mode, workers) row where interp is slower than legacy")
    reg.add_argument("--results", default=str(default_root / "bench-backends"))
    reg.add_argument("--out", default=str(REPO / "scripts/numsim-v2/coverage/perf_regressions.tsv"))
    reg.add_argument("--engine", default="", help="engine commit label written into every row")
    mega = sub.add_parser("mega", help="Mega-MoE perf-budget workloads, legacy vs interp")
    mega.add_argument("--configs", default=",".join(MEGA_CONFIGS))
    mega.add_argument("--mega-modes", default="numsim,racecheck")
    mega.add_argument("--mega-workers", default="1,16,32")
    mega.add_argument("--impls", default="legacy,interp")
    mega.add_argument("--run-timeout", type=float, default=900.0)
    mega.add_argument("--max-load", type=float, default=40.0)
    mega.add_argument("--v2-package", default=None)
    mega.add_argument("--results", default=str(default_root / "bench-backends"))
    mega.add_argument("--force", action="store_true")
    m1 = sub.add_parser("mega-one")
    m1.add_argument("config", choices=list(MEGA_CONFIGS))
    m1.add_argument("mode", choices=["numsim", "racecheck"])
    m1.add_argument("impl", choices=["legacy", "interp"])
    m1.add_argument("workers_one", type=int)
    m1.add_argument("--v2-package", default=None)
    args = parser.parse_args()
    if args.cmd == "mega":
        return cmd_mega(args)
    if args.cmd == "mega-one":
        return cmd_mega_one(args)
    if args.cmd == "regressions":
        return cmd_regressions(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "case":
        return cmd_case(args)
    return cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
