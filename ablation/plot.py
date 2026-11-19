#!/usr/bin/env python3
"""Plot the kda ablation curves (best-so-far speedup vs cumulative tokens).

    python ablation/plot.py            # writes abl-kda-{all,best,mean}.png next to this file

Reads the curve.json copies under ablation/results/kda/ (made by collect.py).
TIRx full / no-KB use batch 3 (abl3-*, no hmz Landlock fence): in batch 1 the
fence broke the NumSim build, so those two cells lost part of their tools.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "kda"
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e6e5e0"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9890"

# (label, runs used for best/mean, color) in fixed categorical order.
CONFIGS = [
    ("TIRx full", ["abl3-kda-full-r3", "abl3-kda-full-r4"], "#2a78d6"),
    ("TIRx no-KB", ["abl3-kda-nokb-r1", "abl3-kda-nokb-r2"], "#eb6834"),
    ("CUDA", ["abl-kda-cuda-r1", "abl-kda-cuda-r2"], "#1baf7a"),
    ("TIRx no-tools", ["abl-kda-notools-r1", "abl-kda-notools-r2"], "#eda100"),
    ("CuTeDSL", ["abl-kda-cutedsl-r1", "abl-kda-cutedsl-r2"], "#e87ba4"),
    ("Triton", ["abl-kda-triton-r1", "abl-kda-triton-r2"], "#008300"),
]


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def steps(name: str) -> tuple[list[float], list[float]]:
    c = load(name)
    xs = [0.0] + [p["total_tokens"] / 1e6 for p in c["points"]] + [c["final_total_tokens"] / 1e6]
    ys = [0.0] + [p["best_so_far"] for p in c["points"]]
    ys.append(ys[-1])
    return xs, ys


def _style(ax) -> None:
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _end_labels(ax, ends: list[list], x_col: float | None = None) -> None:
    """Direct labels at the line ends, nudged apart; ink text beside a colored dot."""
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 0.12:
            ends[i][1] = ends[i - 1][1] + 0.12
    lx = x_col if x_col is not None else max(e[0] for e in ends) + 40
    for _, y, text, color in ends:
        ax.plot([lx - 14], [y], "o", ms=6, color=color)
        ax.text(lx, y, text, va="center", fontsize=10, color=INK)


def plot_all(out: Path) -> None:
    cells = [("full", "TIRx full"), ("nokb", "TIRx no-KB"), ("notools", "TIRx no-tools"),
             ("cutedsl", "CuTeDSL"), ("cuda", "CUDA"), ("triton", "Triton")]  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True, sharey=True)
    for ax, (cell, title) in zip(axes.flat, cells):
        runs = [(f"abl-kda-{cell}-r{i}", f"batch 1 r{i}", BLUE, i) for i in (1, 2)]
        if cell == "full":
            runs += [(f"abl3-kda-full-r{i}", f"batch 3 r{i} (stopped 12.9 h)", GRAY, i) for i in (1, 2)]
            runs += [(f"abl3-kda-full-r{i}", f"batch 3 r{i}", ORANGE, i - 2) for i in (3, 4)]
        if cell == "nokb":
            runs += [(f"abl3-kda-nokb-r{i}", f"batch 3 r{i}", ORANGE, i) for i in (1, 2)]
        for name, label, color, rep in runs:
            xs, ys = steps(name)
            ls = "-" if rep == 1 else (0, (5, 2))
            ax.step(xs, ys, where="post", color=color, lw=2, ls=ls, label=f"{label}: {ys[-1]:.2f}x")
            ax.plot([xs[-1]], [ys[-1]], "o", ms=6, color=color, mec="white", mew=1.5)
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
        _style(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK, handlelength=2.6)
    for ax in axes[1]:
        ax.set_xlabel("cumulative tokens (M)")
    for ax in axes[:, 0]:
        ax.set_ylabel("best passing speedup vs FlashKDA")
    fig.suptitle("KDA forward ablation: best-so-far speedup vs cumulative tokens "
                 "(Opus 5.5 max, 1 day per run)", x=0.01, ha="left", fontsize=13, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor="white")


def plot_best(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ends = []
    for label, names, color in CONFIGS:
        name = max(names, key=lambda n: steps(n)[1][-1])
        xs, ys = steps(name)
        ax.step(xs, ys, where="post", color=color, lw=2, label=f"{label}: {ys[-1]:.2f}x ({name})")
        ax.plot([xs[-1]], [ys[-1]], "o", ms=7, color=color, mec="white", mew=1.5)
        ends.append([xs[-1], ys[-1], f"{label} {ys[-1]:.2f}x", color])
    _end_labels(ax, ends)
    _style(ax)
    ax.set_xlim(0, 1180)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("cumulative tokens (M)")
    ax.set_ylabel("best passing speedup vs FlashKDA")
    ax.set_title("KDA forward: best run per config, best-so-far speedup vs tokens",
                 loc="left", color=INK, fontsize=13)  # fmt: skip
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")


def plot_mean(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ends = []
    for label, names, color in CONFIGS:
        curves = [steps(n) for n in names]
        stop = min(xs[-1] for xs, _ in curves)  # mean only while both runs still run
        grid = np.linspace(0, stop, 1500)
        mean = [np.mean([ys[bisect.bisect_right(xs, x) - 1] for xs, ys in curves]) for x in grid]
        finals = [ys[-1] for _, ys in curves]
        ax.plot(grid, mean, color=color, lw=2,
                label=f"{label}: mean {np.mean(finals):.2f}x (runs {finals[0]:.2f} / {finals[1]:.2f})")  # fmt: skip
        ax.plot([grid[-1]], [mean[-1]], "o", ms=7, color=color, mec="white", mew=1.5)
        ends.append([grid[-1], mean[-1], f"{label} {mean[-1]:.2f}x", color])
    _end_labels(ax, ends)
    _style(ax)
    ax.set_xlim(0, 1020)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("cumulative tokens (M)")
    ax.set_ylabel("best passing speedup vs FlashKDA (mean of 2 runs)")
    ax.set_title("KDA forward: mean of 2 runs per config, best-so-far speedup vs tokens",
                 loc="left", color=INK, fontsize=13)  # fmt: skip
    ax.text(0.0, -0.11, "Each curve ends where the shorter of its two runs ran out of tokens. "
            "TIRx full / no-KB use batch 3 (no sandbox).",
            transform=ax.transAxes, fontsize=9, color=MUTED)  # fmt: skip
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")


def main() -> int:
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})  # fmt: skip
    for kind, fn in (("all", plot_all), ("best", plot_best), ("mean", plot_mean)):
        out = HERE / f"abl-kda-{kind}.png"
        fn(out)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
