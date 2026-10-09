"""Package the harness and build the NumSim engine extension (``numsim_core_py``)."""

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
        return [path for name in listed.split("\0") if name and (path := skills_dir / name).is_file()]
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
