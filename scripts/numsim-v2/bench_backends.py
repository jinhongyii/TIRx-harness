"""Backend performance comparison: legacy vs v2 interp vs v2 codegen (plan 2.3).

Usage (from ``tirx_harness/``, after ``source ../scripts/dev-env.sh``)::

    $PY ../scripts/numsim-v2/bench_backends.py run   [--cases REGEX] [--modes numsim,racecheck,synccheck]
        [--workers 1,8,32] [--repeats 3] [--max-load 40] [--v2-package DIR] [--codegen-cache DIR]
        [--results DIR] [--case-timeout S] [--force]
    $PY ../scripts/numsim-v2/bench_backends.py render [--results DIR] [--json OUT] [--md OUT]
        [--criterion FILE ...] [--legacy-perf FILE]

``run`` measures every canonical case (``tests/numsim/corpus/canonical_cases.py``)
in each mode whose v2 result matches the frozen legacy conformance snapshot
(``tests/conformance/snapshots``), for the variants

    legacy | v2-interp | v2-codegen O1 | v2-codegen O3   x   max_workers in --workers

Each case runs in its own child process (a codegen abort or an OOM loses one
case, not the sweep) and writes ``<results>/<case>.json``; rows already present
are skipped unless ``--force``. Before every case the host load is checked: the
run waits while the 1-minute load average exceeds ``--max-load`` and records the
CPU count, load average and instantaneous idle (tests/CLAUDE.md preflight).

Per run the timed phases are

* v2: ``transpile`` (module cache hit), ``bind`` (``canonicalize_inputs``),
  ``build`` / ``run`` / ``check`` from ``numsim_core_py.run(...)["timing"]``
  (``build`` = codegen emit + cargo freshness check + rustc on a miss + dlopen;
  ~0 for interp), ``report`` (the rest of the Python wall: output decoding,
  payload rendering), and ``engine`` = the whole ``Engine.run`` /
  phase-loop wall clock (bind + native + report). Cold lowering
  (``NUMSIM_V2_NO_CACHE=1``) and a cold codegen build (``gen-*`` cache entries
  removed; the numsim-core rlib stays cached) are measured once per case.
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
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "tirx_harness"
MODES = ("numsim", "racecheck", "synccheck")
VARIANTS = ("legacy", "interp", "codegen-O1", "codegen-O3")
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


def run_v2(v2: Any, entry: Any, mode: str, variant: str, workers: int, expected: Any,
           codegen_cache: Path) -> tuple[dict, dict]:
    from tests.conformance import snapshot as snap

    run_mod = sys.modules[v2.Engine.__module__]
    case = entry.prepare()
    t_tr = Timer()
    with t_tr():
        module = v2.transpile(case.kernel)
    backend, opt = ("interp", 1) if variant == "interp" else ("codegen", int(variant[-1]))
    engine = v2.Engine(max_workers=workers, backend=backend, opt_level=opt)
    native = run_mod.native()
    t_bind, t_native = Timer(), Timer()
    timing = {"build": 0.0, "run": 0.0, "check": 0.0}
    stats: dict[str, Any] = {}

    def native_run(module_, bound, mode_, **extra):
        if "synccheck_limits" in extra:
            extra = {**extra, "synccheck_limits": dict(extra["synccheck_limits"])}
        with t_native():
            raw = native.run(
                module_.handle, {name: b.native for name, b in bound.items()}, mode=mode_,
                backend=engine.backend, workers=engine.max_workers, seed=engine.seed,
                loop_budget=engine.loop_budget, quantum=engine.quantum, opt_level=engine.opt_level,
                codegen_cache_dir=str(codegen_cache), **extra,
            )
        for key in timing:
            timing[key] += float((raw.get("timing") or {}).get(key, 0.0)) / 1e3
        stats.update({k: v for k, v in (raw.get("stats") or {}).items() if isinstance(v, (int, float))})
        return raw

    engine._native_run = native_run
    original_bind = run_mod.canonicalize_inputs

    def timed_bind(*a, **k):
        with t_bind():
            return original_bind(*a, **k)

    run_mod.canonicalize_inputs = timed_bind
    resolver = snap.SourceResolver(module)
    t_eng = Timer()
    try:
        if mode == "numsim":
            with t_eng():
                result = engine.run(module, case.args, subset=case.subset, assumptions=case.assumptions,
                                    outputs=case.outputs)
            ok = bool(v2.compare(result, expected, tolerances=case.comparisons).ok)
            norm = snap.normalize_numsim(result.outputs, result.diagnostics, reference_ok=ok, resolver=resolver)
        else:
            with t_eng():
                payloads = _phases(v2, engine, module, case, entry, mode)
            norm = {"phases": [snap.normalize_analysis_phase(p, resolver) for p in payloads]}
    finally:
        run_mod.canonicalize_inputs = original_bind
    times = {
        "transpile": t_tr.t,
        "engine": t_eng.t,
        "bind": t_bind.t,
        "native": t_native.t,
        "build": timing["build"],
        "run": timing["run"],
        "check": timing["check"],
        "report": max(0.0, t_eng.t - t_bind.t - t_native.t),
    }
    return norm, {"times": times, "stats": stats}


def clear_codegen_libs(cache: Path) -> int:
    removed = 0
    for path in cache.glob("gen-*"):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def bench_case(args: argparse.Namespace) -> dict[str, Any]:
    os.chdir(HARNESS)
    sys.path.insert(0, str(HARNESS))
    v2 = import_v2(args.v2_package)
    from tests.conformance import snapshot as snap
    from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES

    entry = next(e for e in CANONICAL_KERNEL_CASES if e.name == args.case)
    codegen_cache = Path(args.codegen_cache)
    codegen_cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"case": entry.name, "modes": {}, "preflight": [wait_for_quiet(args.max_load)]}
    args.prebuilt = None
    if args.prebuild_dir:
        record = Path(args.prebuild_dir) / f"{entry.name}.json"
        args.prebuilt = json.loads(record.read_text()) if record.exists() else {"builds": {}}
        out["codegen_cold"] = {}
        for variant in ("codegen-O1", "codegen-O3"):
            build = args.prebuilt.get("builds", {}).get(variant, {})
            if "build_s" in build:
                out["codegen_cold"][variant] = {"build_s": build["build_s"], "prebuilt": True,
                                                "source_bytes": (args.prebuilt.get("source_bytes") or [None])[0]}
            elif variant in args.variants:
                # No cached library: a timed run would include a cold build.
                args.variants = [v for v in args.variants if v != variant]
                out.setdefault("codegen_dropped", {})[variant] = build.get("error") or args.prebuilt.get(
                    "error", "no prebuild record")[-500:]

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

    cold_done: set[str] = set()
    workers = [int(w) for w in args.workers.split(",")]
    for mode in args.modes.split(","):
        stored = snap.load_snapshot(entry.name, mode)
        row: dict[str, Any] = {"status": "pending", "runs": {}}
        out["modes"][mode] = row
        if stored is None or "error" in stored:
            row["status"] = "skipped: no legacy oracle"
            continue
        stored = {k: v for k, v in stored.items() if k not in ("schema", "case", "mode")}
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
            relaxed = snap.relax_unanchored(stored, norm)
            if relaxed != stored:
                return f"{variant} does not match the legacy snapshot:\n" + snap.diff_snapshots(stored, relaxed)[:4000]
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
                norm, rec = run_v2(v2, entry, mode, variant, w, expected, codegen_cache)
            problem = check(variant, norm)
            if problem:
                raise RowAbort(problem, norm)
            rec["load1"] = round(os.getloadavg()[0], 2)
            return rec

        try:
            # Eligibility: v2-interp must match the legacy snapshot.
            first = one("interp", workers[0])
            row["eligibility_run"] = first
            # Cold codegen builds (once per case and opt level; the module is
            # mode independent). With a prebuild record (``prebuild``
            # subcommand) the cold times come from there and the cache is
            # already warm; otherwise build inline here. Clear once: O1 and O3
            # libraries have distinct keys and must both stay cached.
            cold_todo = [v for v in ("codegen-O1", "codegen-O3") if v in args.variants and v not in cold_done]
            if cold_todo and args.prebuilt is not None:
                cold_todo = []
            elif cold_todo:
                clear_codegen_libs(codegen_cache)
            for variant in cold_todo:
                rec = one(variant, workers[0])
                out.setdefault("codegen_cold", {})[variant] = {"build_s": rec["times"]["build"], "mode": mode, "run": rec}
                cold_done.add(variant)
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



# --------------------------------------------------------------------------
# Codegen prebuild (cold build times; parallel, before the timed sweep)


def build_case(args: argparse.Namespace) -> dict[str, Any]:
    """Child: build the codegen library of one case at each opt level.

    A ``max_rounds=1`` NumSim run triggers ``backend_for`` (emit + rustc +
    dlopen) and stops right after; ``timing.build`` is the cold build time
    (the numsim-core rlib must already be cached). ``NUMSIM_CODEGEN_LOG=1``
    makes the engine print the cache key, from which the generated source
    size is read.
    """

    os.chdir(HARNESS)
    sys.path.insert(0, str(HARNESS))
    v2 = import_v2(args.v2_package)
    from tests.numsim.corpus.canonical_cases import CANONICAL_KERNEL_CASES

    run_mod = sys.modules[v2.Engine.__module__]
    entry = next(e for e in CANONICAL_KERNEL_CASES if e.name == args.case)
    case = entry.prepare()
    module = v2.transpile(case.kernel)
    bound = run_mod.canonicalize_inputs(module, case.args)
    out: dict[str, Any] = {"case": entry.name, "builds": {}}
    for opt in (1, 3):
        started = time.perf_counter()
        try:
            raw = run_mod.native().run(
                module.handle, {name: b.native for name, b in bound.items()}, mode="numsim",
                backend="codegen", workers=1, max_rounds=1, opt_level=opt,
                codegen_cache_dir=str(args.codegen_cache),
            )
            out["builds"][f"codegen-O{opt}"] = {"build_s": float(raw["timing"]["build"]) / 1e3,
                                                 "wall_s": time.perf_counter() - started}
        except Exception as error:  # noqa: BLE001
            out["builds"][f"codegen-O{opt}"] = {"error": f"{type(error).__name__}: {str(error)[:1500]}",
                                                 "wall_s": time.perf_counter() - started}
    return out


def cmd_prebuild(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    names = selected_cases(args)
    target = Path(args.results) / "prebuild"
    target.mkdir(parents=True, exist_ok=True)
    cache = Path(args.codegen_cache)
    if args.clear:
        print(f"cleared {clear_codegen_libs(cache)} cached libraries", flush=True)

    def one(name: str) -> None:
        path = target / f"{name}.json"
        if path.exists() and not args.force:
            return
        cmd = [sys.executable, __file__, "case-build", name, "--codegen-cache", args.codegen_cache]
        if args.v2_package:
            cmd += ["--v2-package", args.v2_package]
        env = {**os.environ, "NUMSIM_CODEGEN_LOG": "1"}
        started = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.build_timeout, env=env)
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as error:
            stdout = ""
            stderr = (error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or ""))
            stderr += f"\ntimeout after {args.build_timeout}s"
        if "@@RESULT@@" in stdout:
            data = json.loads(stdout.split("@@RESULT@@", 1)[1])
        else:
            data = {"case": name, "error": stderr[-3000:], "builds": {}}
        keys = re.findall(r"key (\w+) emit ([0-9.]+)ms core \S+ rustc (\S+)", stderr)
        data["keys"] = [k for k, _, _ in keys]
        sizes = []
        for key in data["keys"]:
            src = cache / f"gen-{key}" / "lib.rs"
            sizes.append(src.stat().st_size if src.exists() else None)
        data["source_bytes"] = sizes
        data["prebuild_wall_s"] = time.perf_counter() - started
        data["preflight"] = preflight(0.2)
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
        builds = {k: (round(v["build_s"], 1) if "build_s" in v else "ERR") for k, v in data.get("builds", {}).items()}
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {builds} src={sizes} {data.get('error', '')[-200:]}", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(one, names))
    return 0


def cmd_case_build(args: argparse.Namespace) -> int:
    data = build_case(args)
    sys.stdout.write("@@RESULT@@" + json.dumps(data, default=str))
    sys.stdout.flush()
    return 0


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
               "--codegen-cache", args.codegen_cache, "--variants", ",".join(args.variants)]
        prebuild_dir = results / "prebuild"
        if prebuild_dir.exists() and not args.inline_cold_builds:
            cmd += ["--prebuild-dir", str(prebuild_dir)]
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


def cmd_render(args: argparse.Namespace) -> int:
    results = Path(args.results)
    cases = [json.loads(p.read_text()) for p in sorted(results.glob("*.json"))]
    meta = json.loads(Path(args.meta).read_text()) if args.meta and Path(args.meta).exists() else {}
    criterion = {Path(f).stem: Path(f).read_text() for f in args.criterion or ()}
    legacy_perf = json.loads(Path(args.legacy_perf).read_text()) if args.legacy_perf else None
    workers = sorted({w for c in cases for row in c.get("modes", {}).values()
                      for runs in row.get("runs", {}).values() for w in runs}, key=int)
    doc = {"meta": meta, "cases": cases, "legacy_perf_tests": legacy_perf}
    Path(args.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    md: list[str] = []
    add = md.append
    add("---\norphan: true\n---\n")
    add("# NumSim backend comparison (legacy vs v2 interp vs v2 codegen)\n")
    add("Generated by `scripts/numsim-v2/bench_backends.py`; raw data in `backend-comparison.json`. "
        "Plan section 2.3: two backends behind one switch, compare on the corpus per mode, delete the loser.\n")
    if meta:
        add("## Setup\n")
        for key, value in meta.items():
            add(f"- **{key}**: {value}")
        add("")
    add("Times are the minimum over the repetitions of the engine wall clock: legacy `Engine.run` / the "
        "phase loop (artifact already compiled); v2 the same public calls (input binding + native run + "
        "report rendering), with the codegen library **cached** (cold build listed separately). "
        "`core` columns are the v2 native `run + check` only (from `numsim_core_py.run()['timing']`). "
        "Speedup = legacy / variant (> 1 means faster than legacy); `cg/int` = interp / codegen-O1 engine "
        "time (> 1 means codegen faster).\n")

    # Summary.
    add("## Summary\n")
    add("Geometric means over rows measured in every variant (speedup vs legacy, engine wall clock).\n")
    add("| mode | workers | rows | legacy | interp | codegen O1 | codegen O3 | interp vs legacy | O1 vs legacy | O3 vs legacy | O1 vs interp (engine) | O1 vs interp (core) | O3 vs interp (core) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    summary: dict[str, Any] = {}
    for mode in MODES:
        for w in workers:
            rows = [c["modes"][mode] for c in cases if c.get("modes", {}).get(mode, {}).get("status") == "ok"]
            rows = [r for r in rows if all(_engine(r, v, w) for v in VARIANTS)]
            if not rows:
                continue
            g = {v: geomean([_engine(r, v, w) for r in rows]) for v in VARIANTS}
            sp = {v: geomean([_engine(r, "legacy", w) / _engine(r, v, w) for r in rows]) for v in VARIANTS}
            cg_eng = geomean([_engine(r, "interp", w) / _engine(r, "codegen-O1", w) for r in rows])
            cg_core = geomean([(_core(r, "interp", w) or 0) / max(1e-9, _core(r, "codegen-O1", w) or 0) for r in rows
                               if _core(r, "interp", w) and _core(r, "codegen-O1", w)])
            o3_core = geomean([(_core(r, "interp", w) or 0) / max(1e-9, _core(r, "codegen-O3", w) or 0) for r in rows
                               if _core(r, "interp", w) and _core(r, "codegen-O3", w)])
            summary[f"{mode}/{w}"] = {"rows": len(rows), "geomean_s": g, "speedup_vs_legacy": sp,
                                      "codegen_o1_vs_interp_engine": cg_eng, "codegen_o1_vs_interp_core": cg_core,
                                      "codegen_o3_vs_interp_core": o3_core}
            add(f"| {mode} | {w} | {len(rows)} | {fmt_s(g['legacy'])} | {fmt_s(g['interp'])} | "
                f"{fmt_s(g['codegen-O1'])} | {fmt_s(g['codegen-O3'])} | {fmt_x(sp['interp'])} | "
                f"{fmt_x(sp['codegen-O1'])} | {fmt_x(sp['codegen-O3'])} | {fmt_x(cg_eng)} | {fmt_x(cg_core)} | {fmt_x(o3_core)} |")
    add("")

    # Codegen build cost.
    cold = [(c["case"], c["codegen_cold"]) for c in cases if c.get("codegen_cold")]
    if cold:
        add("## Codegen build cost\n")
        b1 = [x["codegen-O1"]["build_s"] for _, x in cold if "codegen-O1" in x]
        b3 = [x["codegen-O3"]["build_s"] for _, x in cold if "codegen-O3" in x]
        warm = [c["modes"][m]["runs"][v][w]["min"]["build"] for c in cases for m in c.get("modes", {})
                for v in ("codegen-O1", "codegen-O3") for w in c["modes"][m].get("runs", {}).get(v, {})]
        lower = [c["lower_cold_s"] for c in cases if "lower_cold_s" in c]
        add("| quantity | n | median | max |")
        add("| --- | --- | --- | --- |")
        for label, xs in (("cold codegen build O1 (rustc miss)", b1), ("cold codegen build O3 (rustc miss)", b3),
                          ("cached codegen build (hit: cargo freshness + dlopen)", warm),
                          ("cold v2 lowering (no module cache)", lower)):
            if xs:
                add(f"| {label} | {len(xs)} | {fmt_s(statistics.median(xs))} | {fmt_s(max(xs))} |")
        add("")

    # Win/loss lists.
    add("## Where each backend wins\n")
    for mode in MODES:
        for w in workers:
            rows = [(c["case"], c["modes"][mode], c.get("instr_mix", {})) for c in cases
                    if c.get("modes", {}).get(mode, {}).get("status") == "ok"]
            pairs = [(n, (_core(r, "interp", w) or 0) / max(1e-9, _core(r, "codegen-O1", w) or 0), mix)
                     for n, r, mix in rows if _core(r, "interp", w) and _core(r, "codegen-O1", w)]
            if not pairs:
                continue
            wins = sorted([p for p in pairs if p[1] > 1.05], key=lambda p: -p[1])
            losses = sorted([p for p in pairs if p[1] < 0.95], key=lambda p: p[1])
            add(f"- **{mode}, {w} workers** (core time, codegen O1 vs interp): codegen faster by >5% on "
                f"{len(wins)}/{len(pairs)}, interp faster by >5% on {len(losses)}/{len(pairs)}. "
                + ("Best codegen: " + ", ".join(f"`{n}` {x:.2f}x (heavy {m.get('heavy_pct', 0):.0f}%)" for n, x, m in wins[:5]) + ". " if wins else "")
                + ("Best interp: " + ", ".join(f"`{n}` {1 / x:.2f}x (heavy {m.get('heavy_pct', 0):.0f}%)" for n, x, m in losses[:5]) + "." if losses else ""))
    add("")

    # Instruction mix correlation.
    add("## Instruction mix\n")
    add("Static `Instr` counts of the lowered module (`numsim_core_py.Module.to_json`), per family. "
        "`heavy` = tile / TMA / bulk copy / tcgen05 / ldmatrix instructions whose single dispatch does a "
        "whole tile of work; the rest are per-lane scalar, control, warp and memory instructions where "
        "dispatch overhead is what codegen removes. Rows bucketed by heavy share, codegen O1 vs interp "
        "core-time geomean at the largest worker count measured:\n")
    w = workers[-1] if workers else None
    add("| mode | bucket | rows | O1 vs interp (core) | O3 vs interp (core) | interp vs legacy (engine) |")
    add("| --- | --- | --- | --- | --- | --- |")
    buckets = (("scalar-heavy (heavy < 2%)", 0, 2), ("mixed (2-8%)", 2, 8), ("tile/TMA/MMA-heavy (>= 8%)", 8, 101))
    for mode in MODES:
        for label, lo, hi in buckets:
            rows = [c["modes"][mode] for c in cases if c.get("modes", {}).get(mode, {}).get("status") == "ok"
                    and lo <= c.get("instr_mix", {}).get("heavy_pct", 0) < hi]
            rows = [r for r in rows if _core(r, "interp", w) and _core(r, "codegen-O1", w)]
            if not rows:
                continue
            o1 = geomean([_core(r, "interp", w) / _core(r, "codegen-O1", w) for r in rows])
            o3 = geomean([_core(r, "interp", w) / _core(r, "codegen-O3", w) for r in rows if _core(r, "codegen-O3", w)])
            il = geomean([_engine(r, "legacy", w) / _engine(r, "interp", w) for r in rows if _engine(r, "legacy", w)])
            add(f"| {mode} | {label} | {len(rows)} | {fmt_x(o1)} | {fmt_x(o3)} | {fmt_x(il)} |")
    add("")

    # Per-case tables.
    for mode in MODES:
        for w in workers:
            rows = [(c, c["modes"][mode]) for c in cases if mode in c.get("modes", {})
                    and c["modes"][mode].get("status") == "ok" and str(w) in c["modes"][mode].get("runs", {}).get("interp", {})]
            if not rows:
                continue
            add(f"## {mode}, max_workers={w}\n")
            add("| case | heavy% | dyn instrs | legacy | interp | O1 | O1 cold build | O3 | O3 cold build | interp/legacy | O1/legacy | O3/legacy | core interp | core O1 | core O3 | cg/int core |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for c, r in sorted(rows, key=lambda cr: cr[0]["case"]):
                leg, it, o1, o3 = (_engine(r, v, str(w)) for v in VARIANTS)
                cold_ = c.get("codegen_cold", {})
                instrs = r["runs"]["interp"][str(w)].get("stats", {}).get("instrs")
                ci, c1, c3 = (_core(r, v, str(w)) for v in ("interp", "codegen-O1", "codegen-O3"))
                add(f"| `{c['case']}` | {c.get('instr_mix', {}).get('heavy_pct', 0):.1f} | {instrs if instrs is not None else '-'} | "
                    f"{fmt_s(leg)} | {fmt_s(it)} | {fmt_s(o1)} | {fmt_s(cold_.get('codegen-O1', {}).get('build_s'))} | "
                    f"{fmt_s(o3)} | {fmt_s(cold_.get('codegen-O3', {}).get('build_s'))} | "
                    f"{fmt_x(leg / it if leg and it else None)} | {fmt_x(leg / o1 if leg and o1 else None)} | "
                    f"{fmt_x(leg / o3 if leg and o3 else None)} | {fmt_s(ci)} | {fmt_s(c1)} | {fmt_s(c3)} | "
                    f"{fmt_x(ci / c1 if ci and c1 else None)} |")
            add("")

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
    if args.recommendation and Path(args.recommendation).exists():
        add(Path(args.recommendation).read_text())
    Path(args.md).write_text("\n".join(md) + "\n")
    print(f"wrote {args.json} and {args.md}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_cache = Path(os.environ.get("NUMSIM_CACHE_DIR", Path.home() / ".cache/tirx-harness/numsim")) / "bench-codegen"

    def common(p):
        p.add_argument("--modes", default=",".join(MODES))
        p.add_argument("--workers", default="1,8,32")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--max-load", type=float, default=40.0)
        p.add_argument("--v2-package", default=None, help="frozen copy of the v2 package (with its .so)")
        p.add_argument("--codegen-cache", default=str(default_cache))
        p.add_argument("--variants", type=lambda s: s.split(","), default=list(VARIANTS))

    run = sub.add_parser("run")
    common(run)
    run.add_argument("--cases", default=".")
    run.add_argument("--results", default=str(default_cache.parent / "bench-backends"))
    run.add_argument("--status-md", default=str(REPO / "docs/development/v2-conformance-status.md"),
                     help="only modes listed as 'match' here ('' = all modes; rows are re-verified anyway)")
    run.add_argument("--case-timeout", type=float, default=3 * 3600)
    run.add_argument("--force", action="store_true")
    run.add_argument("--inline-cold-builds", action="store_true",
                     help="measure cold codegen builds inside the sweep even when <results>/prebuild exists")
    case = sub.add_parser("case")
    case.add_argument("case")
    case.add_argument("--prebuild-dir", default=None)
    common(case)
    pre = sub.add_parser("prebuild", help="build every case's codegen libraries in parallel (cold build times)")
    common(pre)
    pre.add_argument("--cases", default=".")
    pre.add_argument("--results", default=str(default_cache.parent / "bench-backends"))
    pre.add_argument("--status-md", default=str(REPO / "docs/development/v2-conformance-status.md"))
    pre.add_argument("--jobs", type=int, default=12)
    pre.add_argument("--build-timeout", type=float, default=2 * 3600)
    pre.add_argument("--clear", action="store_true", help="remove cached gen-* libraries first")
    pre.add_argument("--force", action="store_true")
    cb = sub.add_parser("case-build")
    cb.add_argument("case")
    common(cb)
    render = sub.add_parser("render")
    render.add_argument("--results", default=str(default_cache.parent / "bench-backends"))
    render.add_argument("--json", default=str(REPO / "docs/development/backend-comparison.json"))
    render.add_argument("--md", default=str(REPO / "docs/development/backend-comparison.md"))
    render.add_argument("--meta", default=None, help="JSON object of setup facts for the header")
    render.add_argument("--criterion", nargs="*", help="criterion output text files (one section each)")
    render.add_argument("--legacy-perf", default=None, help="JSON with a 'markdown' key")
    render.add_argument("--recommendation", default=None, help="markdown appended verbatim")
    args = parser.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "case":
        return cmd_case(args)
    if args.cmd == "prebuild":
        return cmd_prebuild(args)
    if args.cmd == "case-build":
        return cmd_case_build(args)
    return cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
