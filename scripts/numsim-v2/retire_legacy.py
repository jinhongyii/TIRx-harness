"""Redesign step 5: delete the legacy NumSim engine and switch the public names to v2.

Usage (repository root)::

    python scripts/numsim-v2/retire_legacy.py --dry-run            # print the whole plan
    python scripts/numsim-v2/retire_legacy.py --dry-run --list     # ... with every file / test id
    python scripts/numsim-v2/retire_legacy.py --apply              # refuses while blockers remain

The plan has seven parts, printed in this order and applied in this order:

1. ``git rm`` the legacy code: ``engine-rs/``, ``frontend-rs/``, the
   ``thirdparty/tvm-rust-ext`` submodule (and its ``.gitmodules`` entry), the
   legacy Python modules under ``numsim/`` and ``transpiler/``, and the build
   artefacts they leave in the source tree.
2. Keep and relocate what v2 still imports from the legacy layer
   (``errors``, ``cases``, ``dtype_abi`` + ``dtype_registry.json``,
   ``report``; the ``compare`` closure of ``api.py`` moves to
   ``v2/compare.py``) and the one data file a v2 tool reads from
   ``engine-rs/`` (``SUPPORTED_OPS.md`` -> ``numsim-oplib/``).
3. Retire legacy tests: the union of ``retire_tests.py`` waves 0, 1, 2, 4
   and 5b (wave 3, the corpus gates, once conformance matches), plus test
   support modules no surviving test imports. A file whose every test is
   retired is removed; otherwise the functions are cut.
4. Rewrite the files that switch the implementation (generated content):
   ``tirx_harness/__init__.py``, ``numsim/__init__.py``, ``setup.py``,
   ``MANIFEST.in``, ``tests/conftest.py`` (drop the ``NUMSIM_IMPL`` shim),
   ``tests/conformance`` (v2 is the implementation, snapshots are the oracle),
   CI workflows, ``scripts/smoke_wheel.py``, ``gen_registry.py``.
5. Doc edits: submodule instructions and the ``(pending: ...)`` markers that
   resolve on deletion (the others are listed and kept).
6. Migration tooling that only works against the legacy engine.
7. Blockers (from ``coverage/step5_a_status.tsv``, the wave-2/4 holds and
   ``v2_public_status.tsv``); ``--apply`` refuses while any remain unless
   ``--force``.

Nothing here is run by CI. Regenerate the inputs first: ``classify_tests.py``,
a ``NUMSIM_IMPL=v2`` run summarized by ``v2_public_status.py``, and the
internal-surface run that ``step5_a_status.tsv`` records (see
``docs/development/test-migration.md`` "Step 5").
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import retire_tests  # noqa: E402

PKG = "tirx_harness/src/tirx_harness"
NUMSIM = f"{PKG}/numsim"
COVERAGE = HERE / "coverage"

# ----------------------------------------------------------------------------
# 1. Legacy code

LEGACY_TREES = [
    f"{NUMSIM}/engine-rs",
    "tirx_harness/frontend-rs",
    f"{NUMSIM}/transpiler",
]
SUBMODULE = "thirdparty/tvm-rust-ext"
LEGACY_MODULES = [
    f"{NUMSIM}/{name}"
    for name in (
        "api.py",
        "bindings.py",
        "checker_runner.py",
        "checkers.py",
        "checker_report.py",
        "checker_render.py",
        "abi.py",
        "host_abi.py",
        "value_analysis.py",
    )
]
# Untracked build products of the legacy frontend extension (rm -rf, not git rm).
LEGACY_ARTEFACTS = [
    f"{NUMSIM}/_tvm_rust_ext.abi3.so",
    f"{NUMSIM}/_tvm_rust_ext.identity",
    f"{NUMSIM}/_thirdparty_licenses",
]

# ----------------------------------------------------------------------------
# 2. Kept or relocated (v2 imports these)

KEPT_MODULES = {
    f"{NUMSIM}/errors.py": "v2 raises NumSimError / NumSimExecutionError / UnsupportedTIRxError (api, compile, run)",
    f"{NUMSIM}/cases.py": "public TensorMap / Im2col / NumSimCase / ComparisonSpec; v2 run.py decodes numsim.cases.TensorMap images",
    f"{NUMSIM}/dtype_abi.py": "imported by cases.py (dtype_itemsize)",
    f"{NUMSIM}/dtype_registry.json": "read by dtype_abi.py",
    f"{NUMSIM}/report.py": "Mismatch / NumSimReport used by the compare closure",
}
COMPARE_CLOSURE = [
    "NumSimResult",
    "_comparison_index",
    "_comparison_numeric_view",
    "_decode_comparison_array",
    "_normalize_comparison_selector",
    "compare",
]
RELOCATE = {
    f"{NUMSIM}/engine-rs/SUPPORTED_OPS.md": f"{NUMSIM}/core-rs/numsim-oplib/SUPPORTED_OPS.md",
}

# ----------------------------------------------------------------------------
# 4. Rewrites (generated content or exact substitutions)

NUMSIM_INIT = '''"""NumSim: TIRx -> ``Program`` bytecode executed by the Rust engine (``core-rs``).

The public names are the v2 implementation (``tirx_harness.numsim.v2``).
"""

from .cases import ComparisonRegion, ComparisonSpec, Im2col, NumSimCase, TensorMap
from .errors import (
    NumSimBuildError,
    NumSimError,
    NumSimExecutionError,
    UnmodeledTIRxFormError,
    UnsupportedTIRxError,
)
from .v2 import *  # noqa: F403
from .v2 import __all__ as _v2_all

__all__ = sorted(
    {
        *_v2_all,
        "ComparisonRegion",
        "ComparisonSpec",
        "Im2col",
        "NumSimBuildError",
        "NumSimCase",
        "NumSimError",
        "NumSimExecutionError",
        "TensorMap",
        "UnmodeledTIRxFormError",
        "UnsupportedTIRxError",
    }
)
'''

PKG_INIT = '''"""Installable tooling for TIRx kernel development."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

__version__ = version("tirx-harness")

__all__ = ["racecheck", "synccheck"]


def synccheck(kernel: PrimFunc, inputs: dict | None = None, **options):
    """Run synchronization analysis for one concrete TIRx invocation."""
    from .numsim.v2 import synccheck as _synccheck

    return _synccheck(kernel, inputs, **options)


def racecheck(kernel: PrimFunc, inputs: dict | None = None):
    """Run data-race analysis for one concrete TIRx invocation."""
    from .numsim.v2 import racecheck as _racecheck

    return _racecheck(kernel, inputs)
'''

SETUP_PY = '''"""Package the harness and build the NumSim engine extension (``numsim_core_py``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "tirx_harness" / "src" / "tirx_harness" / "numsim" / "core-rs"
SKILLS = ROOT / "skills"
DEPENDENCY_GROUPS = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]


def dependencies(group: str):
    for entry in DEPENDENCY_GROUPS[group]:
        if isinstance(entry, str):
            yield entry
        else:
            yield from dependencies(entry["include-group"])


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


class CoreBuildExt(build_ext):
    """``cargo build -p numsim-py --features extension-module`` (abi3)."""

    def build_extension(self, ext: Extension) -> None:
        env = os.environ.copy()
        env.setdefault("PYO3_PYTHON", sys.executable)
        env.setdefault("CARGO_TARGET_DIR", str(Path(self.build_temp).resolve() / "core-rs"))
        subprocess.run(
            [env.get("CARGO", "cargo"), "build", "--release", "--locked", "-p", "numsim-py",
             "--features", "extension-module"],
            cwd=CORE, env=env, check=True,
        )
        library = {"darwin": "libnumsim_core_py.dylib", "win32": "numsim_core_py.dll"}.get(
            sys.platform, "libnumsim_core_py.so"
        )
        output = Path(self.get_ext_fullpath(ext.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(env["CARGO_TARGET_DIR"]) / "release" / library, output)


def skill_files(skills_dir: Path) -> list[Path]:
    """Return the source files of the skills in ``skills_dir``, excluding fetched references."""
    skill_dirs = sorted(path.parent for path in skills_dir.glob("*/SKILL.md"))
    if (skills_dir.parent / ".git").exists():
        listed = git("ls-files", "-z", "--", *(skill.name for skill in skill_dirs), cwd=skills_dir)
        return [path for name in listed.split("\\0") if name and (path := skills_dir / name).is_file()]
    return [
        path
        for skill in skill_dirs
        for path in sorted(skill.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


class SkillsBuildPy(build_py):
    def run(self) -> None:
        super().run()
        if self.editable_mode:
            return
        destination = Path(self.build_lib) / "tirx_harness" / "_skills"
        shutil.rmtree(destination, ignore_errors=True)
        for source in skill_files(SKILLS):
            target = destination / source.relative_to(SKILLS)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


setup(
    install_requires=list(dependencies("harness")),
    ext_modules=[Extension("tirx_harness.numsim.v2.numsim_core_py", sources=[], py_limited_api=True)],
    cmdclass={"build_ext": CoreBuildExt, "build_py": SkillsBuildPy},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
'''


@dataclass
class Edit:
    path: str
    what: str
    old: str | None = None  # exact substring to replace (None: whole-file content)
    new: str | None = None
    manual: bool = False  # cannot be generated; printed as a checklist item

    def apply(self) -> None:
        target = REPO / self.path
        if self.manual:
            return
        if self.old is None:
            target.write_text(self.new or "")
            return
        text = target.read_text()
        if self.old not in text:
            raise SystemExit(f"{self.path}: expected text not found for: {self.what}")
        target.write_text(text.replace(self.old, self.new or "", 1))


def rewrites() -> list[Edit]:
    edits = [
        Edit(f"{PKG}/__init__.py", "root racecheck/synccheck call numsim.v2", new=PKG_INIT),
        Edit(f"{NUMSIM}/__init__.py", "public numsim names re-export v2 (+ cases, errors)", new=NUMSIM_INIT),
        Edit(f"{NUMSIM}/v2/report.py", "compare: use the relocated closure",
             old="from tirx_harness.numsim.api import compare as legacy_compare",
             new="from .compare import compare as legacy_compare"),
        Edit(f"{NUMSIM}/v2/compare.py", "new file: compare closure moved from numsim/api.py (generated from AST)",
             new=None),  # filled in plan() from api.py
        Edit("setup.py", "build numsim_core_py instead of the tvm-rust-ext frontend; drop the sdist submodule copy",
             new=SETUP_PY),
        Edit("MANIFEST.in", "prune core-rs target instead of engine-rs tests",
             old="prune tirx_harness/src/tirx_harness/numsim/engine-rs/tests",
             new="prune tirx_harness/src/tirx_harness/numsim/core-rs/target"),
        Edit(f"{NUMSIM}/core-rs/numsim-oplib/tools/gen_registry.py", "registry source moved next to oplib",
             old='DEFAULT_SOURCE = CRATE.parents[1] / "engine-rs" / "SUPPORTED_OPS.md"',
             new='DEFAULT_SOURCE = CRATE / "SUPPORTED_OPS.md"'),
        Edit("tirx_harness/tests/conftest.py", "drop the NUMSIM_IMPL=v2 rebinding shim (numsim *is* v2)",
             manual=True),
        Edit("tirx_harness/tests/conformance/snapshot.py",
             "IMPLS = ('v2',); drop delta_snapshot_path (deltas are folded); --update-snapshots regenerates from v2 "
             "(decision: every changed snapshot cites a delta row in the commit message)", manual=True),
        Edit("tirx_harness/tests/conformance/test_conformance.py", "drop the legacy import path and the v2 skip guards",
             manual=True),
        Edit("tirx_harness/tests/conformance/README.md", "v2 is the implementation; regeneration rule", manual=True),
        Edit(".github/workflows/tests.yml", "replace the submodule init step with "
             "`bash tirx_harness/src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh`; add "
             "`cargo test --workspace` (core-rs) and NUMSIM_IMPL-free conformance", manual=True),
        Edit(".github/workflows/build_wheels.yml", "drop the tvm-rust-ext archive download; wheels build numsim_core_py",
             manual=True),
        Edit("scripts/smoke_wheel.py", "check numsim_core_py instead of the tvm-rust-ext licenses", manual=True),
        Edit("tirx_harness/tests/numsim/support/paths.py", "drop ENGINE_ROOT (engine-rs)", manual=True),
        Edit("tirx_harness/tests/numsim/support/manifest.py",
             "delete the legacy-only views (kernel_manifest, resolved_kernel, call_op_names, emitted_calls, "
             "emitted_module); after the cuts no surviving test calls them (check with --list, section 7)", manual=True),
        Edit(f"{NUMSIM}/CLAUDE.md", "remove the legacy-engine paragraph and the legacy snapshot policy", manual=True),
        Edit(f"{NUMSIM}/AGENTS.md", "mirror CLAUDE.md", manual=True),
        Edit(".github/workflows/tests.yml", "add `python scripts/numsim-v2/check_snapshot_deltas.py --base origin/main` "
             "(fails when a snapshot changes without a delta row cited in the commit message)", manual=True),
        Edit(f"{NUMSIM}/v2/run.py", "decision: a failed codegen build raises NumSimBuildError (today ValueError) [W8]", manual=True),
        Edit(f"{NUMSIM}/v2/api.py", "decision: missing bindings report `incomplete` instead of raising InputError [W8]", manual=True),
        Edit(f"{NUMSIM}/core-rs/numsim-oplib/tools/gen_registry.py",
             "decision: SUPPORTED_OPS.md becomes generated from the oplib registry (invert the generator) [W4]", manual=True),
    ]
    return edits


SUBMODULE_DOC_LINES = [
    ("docs/optimization-runs.md", "git submodule update --init thirdparty/tvm-rust-ext"),
    ("docs/installation.md", "git submodule update --init thirdparty/tvm-rust-ext"),
    ("tirx_harness/tests/CLAUDE.md", "git submodule update --init thirdparty/tvm-rust-ext  # repo root"),
    (".github/workflows/tests.yml", "git submodule update --init thirdparty/tvm-rust-ext"),
]

# (pending: ...) markers fall in three classes:
#   resolve - the deletion itself, one of the rewrites above, or a recorded
#             DECISION makes the text true; the parenthetical is removed;
#   action  - an undecided question (none left after the 2026-10-08 decisions);
#   keep    - unrelated to the legacy engine (backend decision, worker count).
RESOLVES = re.compile(r"legacy|delet|submodule|switch|migration completes|numsim_core_py", re.I)
KEEP = re.compile(r"backend decision|detected CPU count", re.I)

# Coordinator decisions for the step-5 open items (2026-10-08). Each names the
# marker text it settles and the change step 5 must make.
DECISIONS = [
    ("snapshot", re.compile(r"snapshot policy|regeneration rule|policy for generating snapshots", re.I),
     "fold every <mode>.delta.json into its base snapshot and delete the delta files (v2 becomes the oracle); "
     "afterwards --update-snapshots regenerates from v2, and every changed snapshot must be justified by a delta "
     "row cited in the commit message (CI: scripts/numsim-v2/check_snapshot_deltas.py)"),
    ("env-prefix", re.compile(r"NUMSIM_V2_. prefix", re.I),
     "rename NUMSIM_V2_* to NUMSIM_*; v2/options.py keeps NUMSIM_V2_* as aliases for one release"),
    ("build-error", re.compile(r"NumSimBuildError", re.I),
     "NumSimBuildError stays: a failed codegen build raises NumSimBuildError (W8 change in v2/run.py)"),
    ("check-failed", re.compile(r"CheckFailed", re.I),
     "CheckFailed stays: reports raise tirx_harness._report.CheckFailed (already the v2 behaviour)"),
    ("ops-table", re.compile(r"table moves", re.I),
     "SUPPORTED_OPS.md moves to numsim-oplib and becomes generated from the oplib registry (W4)"),
    ("multi-launch", re.compile(r"single launch", re.I),
     "multi-launch entry points are the v2 behaviour"),
    ("missing-bindings", re.compile(r"InputError", re.I),
     "missing bindings report `incomplete` (W8 change in the v2 entry points)"),
]


def marker_class(body: str) -> str:
    if KEEP.search(body):
        return "keep"
    for name, pattern, _ in DECISIONS:
        if pattern.search(body):
            return f"resolve:{name}"
    if RESOLVES.search(body):
        return "resolve"
    return "action"


def pending_markers() -> list[tuple[str, int, str, str]]:
    out = []
    for path in sorted(REPO.rglob("*.md")):
        rel = path.relative_to(REPO).as_posix()
        if any(part in rel for part in ("/target/", ".venv", "thirdparty/", "engine-rs/", "frontend-rs/")) or rel.endswith(
            "docs/development/test-migration.md"  # quotes the markers; not a marker itself
        ):
            continue
        text = path.read_text(errors="replace")
        for match in re.finditer(r"\(pending:", text):
            depth, end = 0, match.start()
            for index in range(match.start(), len(text)):
                depth += {"(": 1, ")": -1}.get(text[index], 0)
                if depth == 0:
                    end = index + 1
                    break
            body = " ".join(text[match.start() : end].split())
            line = text.count("\n", 0, match.start()) + 1
            out.append((rel, line, body, marker_class(body)))
    return out


# ----------------------------------------------------------------------------
# 3. Tests


def tracked(path: str) -> list[str]:
    return [p for p in subprocess.run(["git", "ls-files", path], cwd=REPO, capture_output=True, text=True).stdout.split("\n") if p]


def retired_tests(waves: list[str], results: Path | None) -> tuple[set[str], dict[str, str]]:
    ids: set[str] = set()
    info: dict[str, str] = {}
    for wave in waves:
        chosen, extra = retire_tests.selection(wave, results)
        ids |= chosen
        info.update({k: f"wave {wave}: {v}" for k, v in extra.items()})
    return ids, info


def plan_tests(ids: set[str]) -> tuple[list[str], dict[str, list[str]]]:
    per_file: dict[str, set[str]] = collections.defaultdict(set)
    for test_id in ids:
        file, _, func = test_id.partition("::")
        per_file[file].add(func)
    whole, partial = [], {}
    for file, funcs in sorted(per_file.items()):
        path = retire_tests.TESTS_BASE / file
        if not path.exists():
            continue
        present = retire_tests.test_functions(path)
        live = funcs & present.keys()
        if live and live == set(present):
            whole.append(file)
        elif live:
            partial[file] = sorted(live, key=lambda f: present[f][0])
    return whole, partial


def orphaned_support(removed_files: set[str]) -> list[str]:
    """Non-test modules under tests/ that no surviving module imports."""
    base = retire_tests.TESTS_BASE
    modules = {p.relative_to(base).as_posix(): p for p in (base / "tests").rglob("*.py")}
    survivors = {rel for rel in modules if rel not in removed_files}

    def imported_by(rel: str) -> set[str]:
        tree = ast.parse(modules[rel].read_text())
        names = set()
        package = Path(rel).parent.as_posix().split("/")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                parent = package[: len(package) - (node.level - 1)]
                stem = "/".join(parent + (node.module.split(".") if node.module else []))
                names.add(stem + ".py")
                names |= {f"{stem}/{alias.name}.py" for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tests."):
                names.add(node.module.replace(".", "/") + ".py")
                for alias in node.names:
                    names.add(f"{node.module.replace('.', '/')}/{alias.name}.py")
            elif isinstance(node, ast.Import):
                names |= {a.name.replace(".", "/") + ".py" for a in node.names if a.name.startswith("tests.")}
        return names

    used = set()
    for rel in survivors:
        if Path(rel).name.startswith("test_") or Path(rel).name == "conftest.py":
            used |= imported_by(rel)
    # transitive closure through support modules
    frontier = set(used)
    while frontier:
        nxt = set()
        for rel in frontier:
            if rel in modules:
                nxt |= imported_by(rel) - used
        used |= nxt
        frontier = nxt
    return sorted(
        rel for rel in survivors
        if not Path(rel).name.startswith(("test_", "conftest", "__init__"))
        and "/v2/" not in rel and "conformance/" not in rel
        and not rel.startswith("tests/perf/")  # new-layer perf baselines (W8)
        and rel not in used
    )


# ----------------------------------------------------------------------------
# 7. Blockers


DELETED_MODULES = {
    "tirx_harness.numsim." + Path(m).stem for m in LEGACY_MODULES
} | {"tirx_harness.numsim.transpiler"}


def _deleted(mod: str) -> bool:
    return any(mod == d or mod.startswith(d + ".") for d in DELETED_MODULES)


def legacy_import_uses(path: Path, cut: set[str]) -> tuple[dict[str, str], set[str]]:
    """(bound name -> deleted module) imported at any level, and the subset of
    those names still used outside the functions in ``cut``."""
    tree = ast.parse(path.read_text())
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and _deleted(node.module):
            for alias in node.names:
                bound[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tirx_harness.numsim"):
            for alias in node.names:
                if _deleted(f"{node.module}.{alias.name}"):
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _deleted(alias.name):
                    bound[(alias.asname or alias.name).split(".")[0]] = alias.name
    # Live code: module-level statements, surviving test functions, and the
    # helpers / fixtures they reach (by name or by fixture argument).
    defs: dict[str, ast.AST] = {}
    roots: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node
            if node.name.startswith("test_") and node.name not in cut:
                roots.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and f"{node.name}::{sub.name}" not in cut:
                        roots.append(sub)
            else:
                defs[node.name] = node
        elif not isinstance(node, (ast.Import, ast.ImportFrom)):
            roots.append(node)
    used: set[str] = set()
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        node = todo.pop()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                if sub.id in bound:
                    used.add(sub.id)
                elif sub.id in defs and sub.id not in seen:
                    seen.add(sub.id)
                    todo.append(defs[sub.id])
            elif isinstance(sub, ast.arg) and sub.arg in defs and sub.arg not in seen:
                seen.add(sub.arg)
                todo.append(defs[sub.arg])
    return bound, used


def legacy_uses_per_test(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Deleted-module names used at module level (import time), and per test
    function (through the helpers and fixtures it reaches)."""
    tree = ast.parse(path.read_text())
    bound, _ = legacy_import_uses(path, set())
    if not bound:
        return set(), {}
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not (isinstance(n, ast.ClassDef) and n.name.startswith("Test"))}

    def reach(roots: list[ast.AST]) -> set[str]:
        used, seen, todo = set(), set(), list(roots)
        while todo:
            node = todo.pop()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    if sub.id in bound:
                        used.add(sub.id)
                    elif sub.id in defs and sub.id not in seen:
                        seen.add(sub.id)
                        todo.append(defs[sub.id])
                elif isinstance(sub, ast.arg) and sub.arg in defs and sub.arg not in seen:
                    seen.add(sub.arg)
                    todo.append(defs[sub.arg])
        return used

    module_level = reach([n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))])
    tests: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            tests[node.name] = reach([node])
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test_"):
                    tests[f"{node.name}::{sub.name}"] = reach([sub])
    return module_level, tests


def broken_imports(removed: set[str], cut: dict[str, list[str]] | None = None) -> list[str]:
    """Surviving test / support modules whose surviving code still USES a name
    from a deleted legacy module. Unused legacy imports are pruned by --apply."""
    base = retire_tests.TESTS_BASE
    cut = cut or {}
    out = []
    for path in sorted((base / "tests").rglob("*.py")):
        rel = path.relative_to(base).as_posix()
        if rel in removed or "/v2/" in rel:
            continue
        bound, used = legacy_import_uses(path, set(cut.get(rel, ())))
        if used:
            out.append(f"{rel}  ({', '.join(sorted(used))})")
    return out


def prune_legacy_imports(path: Path) -> None:
    """Drop imports of deleted modules whose names nothing uses any more."""
    bound, used = legacy_import_uses(path, set())
    unused = set(bound) - used
    if not unused:
        return
    tree = ast.parse(path.read_text())
    lines = path.read_text().split("\n")
    for node in sorted(ast.walk(tree), key=lambda n: -getattr(n, "lineno", 0)):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = [a for a in node.names if (a.asname or a.name).split(".")[0] in unused or (a.asname or a.name) in unused]
        if names and len(names) == len(node.names):
            del lines[node.lineno - 1 : node.end_lineno]
        elif names:
            keep = [a for a in node.names if a not in names]
            text = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
            indent = lines[node.lineno - 1][: len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
            head = f"from {'.' * node.level}{node.module or ''} import " if isinstance(node, ast.ImportFrom) else "import "
            lines[node.lineno - 1 : node.end_lineno] = [indent + head + text]
    path.write_text("\n".join(lines))


def blockers(info: dict[str, str], removed: set[str] | None = None, cut: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    for entry in broken_imports(removed or set(), cut):
        out["surviving code uses a deleted legacy module"].append(entry)
    status = COVERAGE / "step5_a_status.tsv"
    if status.exists():
        for row in csv.DictReader(status.open(), delimiter="\t"):
            if row["step5"] in {"blocked-v2", "needs-port", "uses-legacy-internals", "ported-held"}:
                out[f"A {row['surface']}: {row['step5']}"].append(row["legacy_test"])
    for test_id, reason in info.items():
        if "hold" in reason and "replacement not passing" in reason:
            out["B/A replacement still xfail (wave 2/4 hold)"].append(test_id)
    return out


# ----------------------------------------------------------------------------


ENV_HELPER = '''def _env(name: str, default: str = "") -> str:
    """``NUMSIM_<name>``; ``NUMSIM_V2_<name>`` is accepted for one release."""
    return os.environ.get(f"NUMSIM_{name}", os.environ.get(f"NUMSIM_V2_{name}", default))'''


@dataclass
class Plan:
    remove: list[str] = field(default_factory=list)
    artefacts: list[str] = field(default_factory=list)
    relocate: dict[str, str] = field(default_factory=dict)
    whole_tests: list[str] = field(default_factory=list)
    cut_tests: dict[str, list[str]] = field(default_factory=dict)
    support: list[str] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)
    doc_lines: list[tuple[str, str]] = field(default_factory=list)
    markers: list[tuple[str, int, str, str]] = field(default_factory=list)
    tooling: list[str] = field(default_factory=list)
    delta_snapshots: list[str] = field(default_factory=list)
    env_files: list[str] = field(default_factory=list)
    blockers: dict[str, list[str]] = field(default_factory=dict)


def compare_module() -> str:
    source = (REPO / NUMSIM / "api.py").read_text()
    tree = ast.parse(source)
    lines = source.split("\n")
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in COMPARE_CLOSURE:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            chunks.append("\n".join(lines[first - 1 : node.end_lineno]))
    header = (
        '"""Output comparison against expected arrays, moved from the legacy ``numsim/api.py``\n'
        '(redesign step 5). Behaviour unchanged."""\n\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass, field\n"
        "from typing import Any\n\n"
        "import numpy as np\n\n"
        "from tirx_harness.numsim.cases import ComparisonRegion, ComparisonSpec\n"
        "from tirx_harness.numsim.errors import NumSimExecutionError\n"
        "from tirx_harness.numsim.report import Mismatch, NumSimReport\n"
    )
    return header + "\n\n" + "\n\n\n".join(chunks) + "\n"


def build(waves: list[str], results: Path | None) -> Plan:
    plan = Plan()
    for tree in LEGACY_TREES + LEGACY_MODULES:
        if (REPO / tree).exists():
            plan.remove.append(tree)
    plan.remove.append(SUBMODULE)
    plan.artefacts = [a for a in LEGACY_ARTEFACTS if (REPO / a).exists()]
    plan.relocate = {s: d for s, d in RELOCATE.items() if (REPO / s).exists()}
    ids, info = retired_tests(waves, results)
    plan.whole_tests, plan.cut_tests = plan_tests(ids)
    plan.support = orphaned_support(set(plan.whole_tests))
    plan.edits = rewrites()
    for edit in plan.edits:
        if edit.path.endswith("v2/compare.py"):
            edit.new = compare_module()
    plan.doc_lines = [(p, line) for p, line in SUBMODULE_DOC_LINES if line in (REPO / p).read_text()]
    plan.markers = pending_markers()
    plan.tooling = [
        "scripts/numsim-v2/capture_plugin.py",  # hooks legacy transpiler entry points
        "scripts/numsim-v2/tile_rejections.py",  # needs a legacy capture run
        "scripts/numsim-v2/lower_sweep.py",  # consumes the legacy capture
    ]
    plan.blockers = blockers(info, set(plan.whole_tests) | set(plan.support), plan.cut_tests)
    plan.delta_snapshots = sorted(
        p.relative_to(REPO).as_posix() for p in (REPO / "tirx_harness/tests/conformance/snapshots").glob("*/*.delta.json")
    )
    plan.env_files = sorted(
        rel for rel in tracked(".")
        if rel.endswith((".py", ".md", ".yml", ".sh", ".toml"))
        and rel not in ("scripts/numsim-v2/retire_legacy.py", "docs/development/test-migration.md")
        and (REPO / rel).is_file()
        and "NUMSIM_V2_" in (REPO / rel).read_text(errors="replace")
    )
    return plan


def count_files(paths: list[str]) -> int:
    return sum(len(tracked(p)) for p in paths)


def show(plan: Plan, listing: bool) -> None:
    def section(title: str) -> None:
        print(f"\n## {title}")

    section("1. git rm legacy code")
    total = 0
    for path in plan.remove:
        n = len(tracked(path)) if path != SUBMODULE else 1
        total += n
        print(f"git rm -r {path}    # {n} tracked file(s)" + ("  (submodule; also edit .gitmodules)" if path == SUBMODULE else ""))
    print(f"rm -rf {' '.join(plan.artefacts)}    # untracked build artefacts" if plan.artefacts else "# no build artefacts present")
    print(f"# {total} tracked files")

    section("2. kept / relocated")
    for path, why in KEPT_MODULES.items():
        print(f"keep   {path}    # {why}")
    print(f"move   {NUMSIM}/api.py::{{{', '.join(COMPARE_CLOSURE)}}} -> {NUMSIM}/v2/compare.py")
    for src, dst in plan.relocate.items():
        print(f"git mv {src} {dst}")

    section("3. legacy tests (waves " + ", ".join(ARGS.waves) + ")")
    n_whole = sum(len(retire_tests.test_functions(retire_tests.TESTS_BASE / f)) for f in plan.whole_tests)
    n_cut = sum(len(v) for v in plan.cut_tests.values())
    print(f"# {len(plan.whole_tests)} test files removed ({n_whole} tests), {n_cut} functions cut from {len(plan.cut_tests)} files")
    print(f"# {len(plan.support)} test support modules no surviving test imports")
    if listing:
        for f in plan.whole_tests:
            print(f"git rm tirx_harness/{f}")
        for f, funcs in plan.cut_tests.items():
            for func in funcs:
                print(f"cut    tirx_harness/{f}::{func}")
    for f in plan.support:
        print(f"git rm tirx_harness/{f}    # orphaned support")

    section("4. rewrites")
    for edit in plan.edits:
        tag = "manual" if edit.manual else ("write " if edit.old is None else "edit  ")
        print(f"{tag} {edit.path}    # {edit.what}")

    section("5. docs")
    for path, line in plan.doc_lines:
        print(f"edit   {path}    # remove `{line.strip()}`")
    kinds = collections.Counter(m[3].split(":")[0] for m in plan.markers)
    print(f"# (pending: ...) markers: {len(plan.markers)} total; " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    for rel, line, body, kind in plan.markers:
        print(f"{kind:22s} {rel}:{line}  {body[:130]}")

    section("5b. decisions (2026-10-08) and the changes they require")
    for name, _, change in DECISIONS:
        print(f"{name:16s} {change}")
    print(f"# fold {len(plan.delta_snapshots)} delta snapshots into their base snapshots:")
    for path in plan.delta_snapshots:
        print(f"fold   {path}")
    print(f"# NUMSIM_V2_* -> NUMSIM_* in {len(plan.env_files)} files (options.py keeps the aliases)")
    for path in plan.env_files:
        print(f"rename {path}")

    section("6. migration tooling that needs the legacy engine")
    for path in plan.tooling:
        print(f"git rm {path}")

    section("7. blockers")
    if not plan.blockers:
        print("# none")
    for kind, tests in sorted(plan.blockers.items()):
        print(f"# {kind}: {len(tests)}")
        if listing:
            for test in sorted(tests):
                print(f"    {test}")


def apply(plan: Plan) -> None:
    for src, dst in plan.relocate.items():
        subprocess.run(["git", "mv", src, dst], cwd=REPO, check=True)
    for edit in plan.edits:
        edit.apply()
    subprocess.run(["git", "rm", "-r", "-q", *[p for p in plan.remove if p != SUBMODULE]], cwd=REPO, check=True)
    subprocess.run(["git", "rm", "-q", SUBMODULE], cwd=REPO, check=True)  # also edits .gitmodules
    for artefact in plan.artefacts:
        subprocess.run(["rm", "-rf", str(REPO / artefact)], check=True)
    subprocess.run(["git", "rm", "-q", *[f"tirx_harness/{f}" for f in plan.whole_tests + plan.support]], cwd=REPO, check=True)
    for file, funcs in plan.cut_tests.items():
        retire_tests.cut(retire_tests.TESTS_BASE / file, funcs)
    for path in (retire_tests.TESTS_BASE / "tests").rglob("*.py"):
        prune_legacy_imports(path)
    for path, line in plan.doc_lines:
        target = REPO / path
        target.write_text(target.read_text().replace(line + "\n", "", 1))
    for rel, _, body, kind in plan.markers:
        if kind.startswith("resolve"):
            target = REPO / rel
            text = target.read_text()
            target.write_text(re.sub(r"\s*\(pending:[^()]*(\([^()]*\)[^()]*)*\)", "", text, count=1)
                              if " ".join(text.split()).count(body) else text)
    for rel in plan.delta_snapshots:
        delta = REPO / rel
        data = json.loads(delta.read_text())
        data.pop("delta", None)
        base = delta.with_name(delta.name.replace(".delta.json", ".json"))
        base.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        subprocess.run(["git", "rm", "-q", rel], cwd=REPO, check=True)
    for rel in plan.env_files:
        target = REPO / rel
        text = target.read_text()
        if rel.endswith("numsim/v2/options.py"):
            text = re.sub(r'os\.environ\.get\("NUMSIM_V2_(\w+)"', r'_env("\1"', text)
            text = text.replace("NUMSIM_V2_", "NUMSIM_")
            text = text.replace("def ", ENV_HELPER + "\n\ndef ", 1)
        else:
            text = text.replace("NUMSIM_V2_", "NUMSIM_")
        target.write_text(text)
    subprocess.run(["git", "rm", "-q", *plan.tooling], cwd=REPO, check=True)
    print("applied; now do the `manual` items, rebuild numsim_core_py, run the suite and conformance", file=sys.stderr)


def main() -> int:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--list", action="store_true", help="print every test id and blocker")
    parser.add_argument("--waves", nargs="+", default=["0", "1", "2", "4", "5b"])
    parser.add_argument("--v2-results", type=Path, help="junit XML of the v2 replacements (waves 2, 4)")
    parser.add_argument("--force", action="store_true", help="apply despite blockers")
    ARGS = parser.parse_args()
    plan = build(ARGS.waves, ARGS.v2_results)
    show(plan, ARGS.list)
    if ARGS.apply:
        if plan.blockers and not ARGS.force:
            print("\nrefusing: blockers remain (see section 7); --force to override", file=sys.stderr)
            return 1
        apply(plan)
    return 0


ARGS: argparse.Namespace

if __name__ == "__main__":
    sys.exit(main())
