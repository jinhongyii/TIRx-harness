"""Render the shared KDA prompt blocks and task-selected language contract."""

from __future__ import annotations

import re
from pathlib import Path

from evolution.preparation.declare import Task, Toolset

_PROMPTS_DIR = Path(__file__).parent / "rules"
_SLOT_RE = re.compile(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}")


def render_file(path: Path, /, **slots: str) -> str:
    """Replace ``{{slot}}`` markers and reject any unfilled marker."""
    text = path.read_text()
    for key, value in slots.items():
        text = text.replace("{{" + key + "}}", value)
    leftover = sorted(set(_SLOT_RE.findall(text)))
    if leftover:
        raise KeyError(f"prompt template {path.name!r}: unfilled slots {leftover}")
    return text


def render(name: str, /, **slots: str) -> str:
    return render_file(_PROMPTS_DIR / f"{name}.md", **slots)


def kernel_authoring_contract(task: Task) -> str:
    if task.kernel_authoring == "TIRx-lite":
        return render("rule_tirx_lite_authoring").rstrip("\n")
    if task.kernel_authoring == "task":
        return (
            "The selected task specification is the sole authority for the "
            "implementation language. Follow it exactly."
        )
    raise ValueError(f"unsupported kernel authoring mode: {task.kernel_authoring!r}")


def _uses_our_tools(task: Task, toolset: Toolset) -> bool:
    """Our tools (NumSim, Synccheck, Racecheck, dump-source) only apply to TIRx-lite."""
    return task.kernel_authoring == "TIRx-lite" and toolset.our_tools


def _require_knowledge_base_for_task_language(task: Task, toolset: Toolset) -> None:
    if task.kernel_authoring == "task" and not toolset.knowledge_base:
        raise ValueError("a task-defined language requires the knowledge base")


def kernel_search_references(task: Task, toolset: Toolset) -> str:
    _require_knowledge_base_for_task_language(task, toolset)
    if task.kernel_authoring == "TIRx-lite" and not toolset.knowledge_base:
        return """\
The installed `tirx_kernels.tirx_lite` package and the TIRx compiler sources
are the reference for the language; read them directly when needed."""
    if task.kernel_authoring == "TIRx-lite":
        return """\
One deliberate source of invention: survey the PTX ISA manual in
`tirx-wiki` broadly, not only the sections the current candidate already
relies on. An instruction or feature you had not considered may suggest a
different structure for the kernel, or a cheaper way to do a local step.
Both kinds of ideas are worth pursuing; do not treat this as a search for
micro-optimizations only.

Study the canonical kernels and their notes in `tirx-wiki` for optimization
ideas and tactics that transfer to this workload."""
    if task.kernel_authoring == "task":
        return """\
Use `KernelWiki` and the selected implementation stack's authoritative
documentation to study Blackwell optimization patterns, permitted canonical
examples, and PTX/GPU semantics. Transfer tactics only when they comply with
the task's implementation-language and own-source contracts; do not substitute
a kernel written in another authoring stack."""
    raise ValueError(f"unsupported kernel authoring mode: {task.kernel_authoring!r}")


def kernel_research_debugging_contract(task: Task, toolset: Toolset) -> str:
    _require_knowledge_base_for_task_language(task, toolset)
    if task.kernel_authoring not in ("TIRx-lite", "task"):
        raise ValueError(f"unsupported kernel authoring mode: {task.kernel_authoring!r}")
    bullets = []
    if task.kernel_authoring == "TIRx-lite" and toolset.knowledge_base:
        bullets.append(
            "- Proactively use `tirx-wiki` to research relevant TIRx-lite/TIRx APIs, canonical\n"
            "  kernels, optimization guidance, and PTX/GPU semantics."
        )
    elif task.kernel_authoring == "task":
        bullets.append(
            "- Proactively use `KernelWiki` to research the selected implementation stack,\n"
            "  permitted canonical examples, optimization guidance, and PTX/GPU semantics."
        )
    if _uses_our_tools(task, toolset):
        bullets.append(
            """\
- For each major optimization direction, invoke `tirx-profile-kernel`; benchmark
  and profile with IKET, add NCU when needed, preserve artifacts, and rerun after
  the change.
- Invoke `tirx-debug-kernel` for failures, suspect results, or changes affecting
  synchronization, cross-thread-visible memory, or numerical behavior. Run the
  applicable TIRx correctness tools and the normal benchmark correctness check.
- If a performance change affects synchronization or shared state, use both
  operational skills. Reading guides or suggesting commands is not evidence."""
        )
    else:
        bullets.append(
            """\
- For each major optimization direction, benchmark and profile the authored
  kernel directly with IKET. Add `ncu-report-skill` and NCU when hardware
  counters are needed. Preserve artifacts and rerun after the change.
- Use Compute Sanitizer and the normal benchmark correctness check for
  failures, suspect results, or changes affecting synchronization,
  cross-thread-visible memory, or numerical behavior."""
        )
    return "\n".join(bullets)


def kernel_remote_gpu_work(task: Task, toolset: Toolset) -> str:
    if _uses_our_tools(task, toolset):
        return (
            "your own correctness runs and timing experiments, the IKET and NCU\n"
            "profiles `tirx-profile-kernel` prescribes, device-side debugging such as\n"
            "Compute Sanitizer"
        )
    if task.kernel_authoring in ("TIRx-lite", "task"):
        return (
            "your own correctness runs and timing experiments, direct IKET profiles,\n"
            "NCU profiles requested through `ncu-report-skill`, device-side debugging\n"
            "such as Compute Sanitizer"
        )
    raise ValueError(f"unsupported kernel authoring mode: {task.kernel_authoring!r}")


def kernel_remote_local_checks(task: Task, toolset: Toolset) -> str:
    if _uses_our_tools(task, toolset):
        return "Pre-GPU checks (Synccheck, Racecheck, NumSim) need no GPU and stay local."
    if task.kernel_authoring in ("TIRx-lite", "task"):
        return "Checks that do not execute CUDA work stay local."
    raise ValueError(f"unsupported kernel authoring mode: {task.kernel_authoring!r}")


def shared_rule_blocks(toolset: Toolset) -> dict[str, str]:
    """Render the banned-read and GPU rules for KDA."""
    if toolset.banned_paths:
        banned = "\n".join(f"   - {path}" for path in toolset.banned_paths)
    else:
        banned = "   (none)"

    return {
        "banned_reads": render("rule_banned_reads", banned_paths_block=banned).rstrip("\n"),
        "gpu": (_PROMPTS_DIR / "rule_gpu.md").read_text().rstrip("\n"),
    }
