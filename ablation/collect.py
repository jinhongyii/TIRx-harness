#!/usr/bin/env python3
"""Collect the token-vs-speedup curve of ablation runs.

    python ablation/collect.py kda_flow_runs/abl-kda-full-r1 [...]  # -> <run>/curve.json
    python ablation/collect.py --plot out.png kda_flow_runs/abl-*   # plus a plot per task

Tokens come from the Claude Code transcripts of every session the run started
(Humanize ralph loop: one fresh session per round, plus any subagents), found
under ~/.hmz/epics/<encoded worktree>/*/sessions/claude/projects/. Each assistant message is
counted once (by message id). ``total`` = input + cache creation + cache read
+ output tokens; ``output`` is reported separately.

Speedup comes from the run worktree's Git history: every commit that changes
frontier/index.json is a point, valued at the best ``speedup`` recorded in that
index. Points are placed at the cumulative token count reached at commit time.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"
HMZ_EPICS = Path.home() / ".hmz" / "epics"


def _encoded(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(path.resolve()))


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _transcripts(worktree: Path) -> list[Path]:
    return sorted(
        (HMZ_EPICS / _encoded(worktree)).glob("*/sessions/claude/projects/**/*.jsonl")
    ) or sorted((PROJECTS / _encoded(worktree)).rglob("*.jsonl"))


def token_events(worktree: Path) -> list[tuple[float, dict]]:
    """(timestamp, usage) for every distinct assistant message of the run."""
    # Humanize keeps each session's Claude config under its epic directory.
    transcripts = _transcripts(worktree)
    usage_by_id: dict[str, tuple[float, dict]] = {}
    for transcript in transcripts:
        for line in transcript.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = row.get("message")
            if row.get("type") != "assistant" or not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not usage or "timestamp" not in row:
                continue
            key = message.get("id") or row.get("uuid")
            # Streaming writes several rows per message; the last carries final usage.
            usage_by_id[key] = (_ts(row["timestamp"]), usage)
    return sorted(usage_by_id.values(), key=lambda item: item[0])


# Executed (not merely read about) tool invocations, counted per Bash command.
TOOL_PATTERNS = {
    "numsim": re.compile(r"numsim", re.I),
    "synccheck": re.compile(r"synccheck", re.I),
    "racecheck": re.compile(r"racecheck", re.I),
    "dump_kernel": re.compile(r"dump_kernel|dump-source"),
    "iket": re.compile(r"\biket\b", re.I),
    "ncu": re.compile(r"\bncu\b"),
    "compute_sanitizer": re.compile(r"compute-sanitizer"),
}
READ_ONLY = re.compile(r"\s*(cat|sed|grep|rg|ls|head|tail|find|wc|less)\b")


def tool_use(worktree: Path) -> dict[str, int]:
    """Skill calls and executed tool commands across the run's sessions."""
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for transcript in _transcripts(worktree):
        for line in transcript.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = row.get("message")
            if row.get("type") != "assistant" or not isinstance(message, dict):
                continue
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                if block.get("id") in seen:
                    continue
                seen.add(block.get("id"))
                tool_input = block.get("input") or {}
                if block.get("name") == "Skill":
                    key = f"skill:{tool_input.get('skill')}"
                    counts[key] = counts.get(key, 0) + 1
                elif block.get("name") == "Bash":
                    command = str(tool_input.get("command", ""))
                    if READ_ONLY.match(command):
                        continue
                    for name, pattern in TOOL_PATTERNS.items():
                        if pattern.search(command):
                            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def cumulative(events: list[tuple[float, dict]]) -> list[dict]:
    total = output = 0
    out = []
    for ts, usage in events:
        output += usage.get("output_tokens", 0) or 0
        total += sum(
            usage.get(k, 0) or 0
            for k in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            )
        )
        out.append({"t": ts, "total": total, "output": output})
    return out


def _speedups(node) -> list[float]:
    """Speedups recorded anywhere in an agent-written index.

    Agents name the field themselves (``speedup``, ``speedup_geomean``, ...).
    Within one mapping, a geomean-style field wins over other speedup fields.
    """
    found = []
    if isinstance(node, dict):
        numeric = {
            k: float(v)
            for k, v in node.items()
            if "speedup" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        geo = [v for k, v in numeric.items() if "geo" in k.lower()]
        if geo:
            # A candidate's geomean; its per-case speedups below must not count.
            return geo
        found.extend(numeric.values())
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                found.extend(_speedups(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_speedups(value))
    return found


def frontier_points(worktree: Path, workload_dir: str) -> list[dict]:
    index = f"{workload_dir}/frontier/index.json"
    log = subprocess.run(
        ["git", "-C", str(worktree), "log", "--reverse", "--format=%H %ct", "--", index],
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    points = []
    for line in filter(None, log):
        sha, ct = line.split()
        shown = subprocess.run(
            ["git", "-C", str(worktree), "show", f"{sha}:{index}"], capture_output=True, text=True
        )
        try:
            speedups = _speedups(json.loads(shown.stdout))
        except json.JSONDecodeError:
            continue
        if speedups:
            points.append({"t": float(ct), "commit": sha, "best_speedup": max(speedups)})
    return points


def collect(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    worktree = run_dir / "worktree"
    tokens = cumulative(token_events(worktree))
    points = frontier_points(worktree, manifest["workload_dir"])
    best = 0.0
    for point in points:
        reached = [e for e in tokens if e["t"] <= point["t"]]
        point["total_tokens"] = reached[-1]["total"] if reached else 0
        point["output_tokens"] = reached[-1]["output"] if reached else 0
        best = max(best, point["best_speedup"])
        point["best_so_far"] = best
    curve = {
        "run": run_dir.name,
        "task": manifest["task"],
        "toolset": manifest["toolset"],
        "final_total_tokens": tokens[-1]["total"] if tokens else 0,
        "final_output_tokens": tokens[-1]["output"] if tokens else 0,
        "tool_use": tool_use(worktree),
        "points": points,
    }
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    return curve


def plot(curves: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({c["task"].split("_cuda")[0].split("_triton")[0].split("_cutedsl")[0]
                       for c in curves})  # fmt: skip
    fig, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 5), squeeze=False)
    for ax, family in zip(axes[0], families):
        for c in curves:
            if not c["task"].startswith(family):
                continue
            xs = [0] + [p["total_tokens"] for p in c["points"]] + [c["final_total_tokens"]]
            best = [p["best_so_far"] for p in c["points"]]
            ys = [0] + best + [best[-1] if best else 0]
            ax.step([x / 1e6 for x in xs], ys, where="post", label=c["run"])
        ax.set_title(family)
        ax.set_xlabel("cumulative tokens (M)")
        ax.set_ylabel("best passing speedup")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    curves = [collect(run) for run in args.runs]
    for c in curves:
        last = c["points"][-1]["best_so_far"] if c["points"] else None
        print(
            f"{c['run']}: {c['final_total_tokens'] / 1e6:.1f}M tokens "
            f"({c['final_output_tokens'] / 1e6:.2f}M output), "
            f"{len(c['points'])} frontier commits, best speedup {last}\n"
            f"    tools: {c['tool_use']}"
        )
    if args.plot:
        plot(curves, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
