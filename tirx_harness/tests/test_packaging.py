"""Build commands must build the NumSim engine extension and carry the skills."""

import runpy
import subprocess
from pathlib import Path

import setuptools
from setuptools import Distribution, Extension


def test_editable_core_extension_is_copied_to_source(tmp_path, monkeypatch):
    """Post-deletion setup.py builds only ``numsim_core_py`` (no legacy frontend
    extension, identity file or tvm-rust-ext licenses); an editable build copies
    it next to the v2 package."""
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    definitions = runpy.run_path(str(Path(__file__).resolve().parents[2] / "setup.py"))
    assert "RustBuildExt" not in definitions
    source = tmp_path / "src"
    package = source / "tirx_harness" / "numsim" / "v2"
    package.mkdir(parents=True)
    extension = Extension("tirx_harness.numsim.v2.numsim_core_py", sources=[], py_limited_api=True)
    distribution = Distribution({
        "packages": ["tirx_harness.numsim.v2"],
        "package_dir": {"": str(source)},
        "ext_modules": [extension],
    })
    command = definitions["CoreBuildExt"](distribution)
    command.build_lib = str(tmp_path / "build")
    command.ensure_finalized()
    library = Path(command.get_ext_fullpath(extension.name))
    library.parent.mkdir(parents=True)
    library.write_bytes(b"numsim core")

    # Setuptools' editable build copies extensions from build_lib to src.
    command.copy_extensions_to_source()

    assert (package / library.name).read_bytes() == library.read_bytes()
    assert not list(package.parent.glob("_tvm_rust_ext*"))


def _setup_definitions(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "setup.py"))


def test_skill_files_in_a_checkout_skip_fetched_references(tmp_path, monkeypatch):
    definitions = _setup_definitions(monkeypatch)
    tracked = [
        "skills/tirx-wiki/SKILL.md",
        "skills/tirx-wiki/references/repos/INDEX.md",
        "skills/tirx-wiki/scripts/fetch_references.py",
    ]
    for relative in tracked:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(relative)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    fetched = tmp_path / "skills/tirx-wiki/references/repos/cutlass/README.md"
    fetched.parent.mkdir(parents=True)
    fetched.write_text("fetched")

    files = definitions["skill_files"](tmp_path / "skills")

    assert sorted(path.relative_to(tmp_path).as_posix() for path in files) == tracked
