#!/usr/bin/env python3
"""Prepare and launch ablation runs, each in its own tmux session.

    # prepare only (evolve init), e.g. the Phase 1 smoke runs
    python ablation/launch.py init --prefix smoke --tasks kda_forward
    # start Humanize ralph loops on prepared runs
    python ablation/launch.py start --prefix smoke --tasks kda_forward --duration PT1H
    # a main batch: replicate 1 of every cell
    python ablation/launch.py init  --prefix abl --rep 1
    python ablation/launch.py start --prefix abl --rep 1 --duration P1D

    # without the hmz Landlock fence (needed for NumSim on this 5.15 host)
    python ablation/launch.py start --prefix abl3 --cells full nokb --rep 1 --flow open

Each run lives in kda_flow_runs/<name>/; the loop's console output goes to
kda_flow_runs/<name>/hmz.log and its start/end times to launch.json.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import AGENT, CELLS, HOME_BANS, REMOTE, TASKS, run_name, task_for  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "kda_flow_runs"
HMZ = Path(sys.executable).parent / "hmz"
MAX_CONCURRENT = 12
FLOWS = {"ralph": "ralph_loop", "open": str(REPO / "ablation" / "flows" / "ralph_open")}
# Every cache a run's tools write, kept inside the run's gitignored .local/ so
# runs never share compiled artifacts or temp files.
CACHE_ENV = {
    "TMPDIR": ".local/tmp",
    "XDG_CACHE_HOME": ".local/cache",
    "TRITON_CACHE_DIR": ".local/cache/triton",
    "CUDA_CACHE_PATH": ".local/cache/nv",
    "TORCHINDUCTOR_CACHE_DIR": ".local/cache/inductor",
    "CUTE_DSL_CACHE_DIR": ".local/cache/cutedsl",
}
TEMP_RULE = """

## Temporary files

Put every scratch or temporary file under `.local/tmp/` in your worktree (create
subdirectories as you like); `TMPDIR` already points there, and tool caches live
under `.local/cache/`. Do not write to `/tmp` or other system temp locations
directly. Do not read, list, or open temporary files or directories that already
existed before you created them, whether in `/tmp`, `.local/tmp/`, or any cache
directory: they may belong to other runs. Only use temporary files that your own
commands created in this run.
"""


def _selected(args) -> list[tuple[str, str, str]]:
    out = []
    for task in args.tasks:
        for cell in args.cells:
            out.append((task, cell, run_name(args.prefix, task, cell, args.rep)))
    return out


def _init_one(task: str, cell: str, name: str) -> str:
    if (RUNS / name).exists():
        return f"{name}: exists, skipped"
    log = REPO / "ablation" / "logs" / f"init-{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "uv", "run", "--package", "tirx-evolution", "evolve", "init",
        "--task", task_for(task, cell), "--toolset", CELLS[cell][1],
        "--remote", REMOTE, "--name", name,
        *[arg for ban in HOME_BANS for arg in ("--ban", ban)],
    ]  # fmt: skip
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    prompt = RUNS / name / "PROMPT.md"
    if result.returncode == 0 and prompt.is_file():
        prompt.write_text(prompt.read_text().rstrip("\n") + "\n" + TEMP_RULE)
    return f"{name}: init exit {result.returncode} (log {log})"


def _running_sessions() -> set[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True
    )
    return set(result.stdout.split()) if result.returncode == 0 else set()


def _start_one(name: str, duration: str, flow: str) -> str:
    run_dir = RUNS / name
    if not (run_dir / "PROMPT.md").is_file():
        return f"{name}: not prepared"
    if name in _running_sessions():
        return f"{name}: already running"
    caches = " ".join(f"{k}=$PWD/{v}" for k, v in CACHE_ENV.items())
    script = (
        # hmz makes each agent's scratch TMPDIR under $TMPDIR; the login default sits
        # under the banned home directory, so keep it inside the (gitignored) .local/.
        f"cd {shlex.quote(str(run_dir / 'worktree'))} && . .venv/bin/activate && "
        f"export {caches} && mkdir -p {' '.join(CACHE_ENV.values())} && "
        f"{shlex.quote(str(HMZ))} exec -f {shlex.quote(FLOWS[flow])} -a agent={AGENT} "
        f"-p budget.duration={duration},budget.graceful=false \"$(cat ../PROMPT.md)\" 2>&1 | tee -a ../hmz.log; "
        f"{shlex.quote(sys.executable)} -c \"import json,time,pathlib;"
        f"p=pathlib.Path('../launch.json');d=json.loads(p.read_text());"
        f"d['ended_at']=time.time();p.write_text(json.dumps(d,indent=2))\""
    )
    (run_dir / "launch.json").write_text(
        json.dumps(
            {"agent": AGENT, "duration": duration, "flow": flow, "started_at": time.time()},
            indent=2,
        )
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "bash", "-lc", script], check=True)
    return f"{name}: started (tmux attach -t {name})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "start"])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--duration", default="P1D", help="ISO-8601 budget, e.g. PT1H, P1D")
    parser.add_argument("--jobs", type=int, default=3, help="parallel evolve init processes")
    parser.add_argument(
        "--flow", choices=FLOWS, default="ralph", help="open: ralph loop without the hmz fence"
    )
    args = parser.parse_args()

    runs = _selected(args)
    if args.action == "init":
        with ThreadPoolExecutor(args.jobs) as pool:
            for line in pool.map(lambda r: _init_one(*r), runs):
                print(line, flush=True)
        return 0

    active = {s for s in _running_sessions() if (RUNS / s).is_dir()}
    if len(active | {name for *_, name in runs}) > MAX_CONCURRENT:
        print(f"refusing: would exceed {MAX_CONCURRENT} concurrent runs (active: {sorted(active)})")
        return 1
    for *_, name in runs:
        print(_start_one(name, args.duration, args.flow), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
