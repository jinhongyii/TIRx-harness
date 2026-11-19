"""The ablation matrix: which task variant and toolset each cell uses."""

from __future__ import annotations

from pathlib import Path

# cell -> (task suffix, toolset)
CELLS: dict[str, tuple[str, str]] = {
    "full": ("", "kda_flow"),  # 1. TIRx-lite, our tools, KB
    "notools": ("", "ablation_no_tools"),  # 2. TIRx-lite, KB only
    "nokb": ("", "ablation_no_kb"),  # 3. TIRx-lite, our tools only
    "cutedsl": ("_cutedsl", "ablation_no_tools"),  # 5. CuTeDSL, KB
    "cuda": ("_cuda", "ablation_no_tools"),  # 6. CUDA (+ inline PTX), KB
    "triton": ("_triton", "ablation_no_tools"),  # 7. Triton, KB
}
TASKS = ("kda_forward", "mla_dsv4")
SHORT = {"kda_forward": "kda", "mla_dsv4": "mla"}

AGENT = "claude/claude-opus-5-5:max"
REMOTE = "http://127.0.0.1:9000"


def run_name(prefix: str, task: str, cell: str, rep: int | None = None) -> str:
    name = f"{prefix}-{SHORT[task]}-{cell}"
    return name if rep is None else f"{name}-r{rep}"


def task_for(task: str, cell: str) -> str:
    return task + CELLS[cell][0]

# Directories outside the run that hold prior KDA/MLA solutions or other runs'
# agent transcripts. Banned by name rather than banning the whole home
# directory, which blocked the agent's own session files and $TMPDIR.
HOME_BANS = tuple(
    str(Path.home() / name)
    for name in ("KDA-internal", "kda-for-kda-release", "experiments", ".hmz")
)
