"""Remote-only tool runners, uploaded as source; this is not a local CLI.

A separate file keeps the uploaded code easy to lint and test.
"""

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"expected a relative file path: {name!r}")
    return path


def unpack_inputs(archive, workdir):
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            path = relative_path(member.name)
            if not member.isfile():
                raise ValueError(f"input is not a regular file: {member.name!r}")
            destination = workdir.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            destination.chmod(0o700 if member.mode & 0o111 else 0o600)


def environment(workdir, overrides):
    return {**os.environ, **overrides, "KCORAL_DIR": str(workdir)}


def _worker_python(application):
    """Keep bare Python commands in the benchmark worker's environment."""
    if application and application[0] in ("python", "python3"):
        return [sys.executable, *application[1:]]
    return application


def collect_files(workdir, paths):
    files, missing = {}, []
    for name in paths:
        path = workdir.joinpath(*relative_path(name).parts)
        if path.is_symlink() or not path.resolve().is_relative_to(workdir.resolve()):
            raise ValueError(f"artifact escapes input directory or is a symlink: {path}")
        entries = sorted(path.rglob("*")) if path.is_dir() else [path]
        found = False
        for entry in entries:
            if entry.is_symlink() or not entry.resolve().is_relative_to(workdir.resolve()):
                raise ValueError(f"artifact escapes input directory or is a symlink: {entry}")
            if entry.is_dir():
                continue
            if not entry.is_file():
                continue
            files[entry.relative_to(workdir).as_posix()] = entry.read_bytes()
            found = True
        if not found:
            missing.append(name)
    return files, missing


def _resolve_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} is not installed in the remote server environment")
    return os.path.abspath(executable)


def _run_command(archive, command, fetch, overrides):
    executable = _resolve_executable(command[0])
    with tempfile.TemporaryDirectory(prefix="kcoral_tool_") as root:
        workdir = Path(root)
        unpack_inputs(archive, workdir)
        completed = subprocess.run(
            [executable, *command[1:]],
            cwd=workdir,
            env=environment(workdir, overrides),
            stdin=subprocess.DEVNULL,
        )
        files, missing = collect_files(workdir, fetch)
        return {"returncode": completed.returncode, "files": files, "missing": missing}


def python(archive, arguments, overrides):
    return _run_command(archive, [sys.executable, *arguments], [], overrides)


def compute_sanitizer(archive, arguments, fetch, overrides):
    return _run_command(archive, ["compute-sanitizer", *arguments], fetch, overrides)


def iket(archive, arguments, overrides):
    executable = _resolve_executable("run-iket")
    boundary = arguments.index("--")
    arguments = [*arguments[: boundary + 1], *_worker_python(arguments[boundary + 1 :])]
    with tempfile.TemporaryDirectory(prefix="kcoral_iket_") as root:
        workdir, reports = Path(root) / "inputs", Path(root) / "reports"
        workdir.mkdir()
        unpack_inputs(archive, workdir)
        child_env = environment(workdir, overrides)
        child_env.setdefault("TVM_IKET_OFFICIAL_PROFILE", "cutlass-4.6.0")
        completed = subprocess.run(
            [executable, "--output-dir", str(reports), *arguments],
            cwd=workdir,
            env=child_env,
            stdin=subprocess.DEVNULL,
        )
        paths = [p.name for p in sorted(reports.iterdir())] if reports.is_dir() else []
        files, _ = collect_files(reports, paths)
        return {
            "returncode": completed.returncode,
            "files": files,
            "missing": [] if files else ["IKET output"],
        }


def _pinned_nvrtc(overrides):
    """Preload the worker's pip NVRTC so the profiled process compiles like the benchmark.

    NCU's injection makes a later dlopen of libnvrtc.so.13 resolve to the system
    CUDA toolkit's NVRTC even when the pip library comes first on
    LD_LIBRARY_PATH, so the profiled process would build different SASS than
    the benchmark. A preloaded library with that soname wins over both.
    """
    if "LD_PRELOAD" in overrides:
        return {}
    try:
        import nvidia
    except ImportError:
        return {}
    for root in nvidia.__path__:
        library = Path(root) / "cu13" / "lib" / "libnvrtc.so.13"
        if library.is_file():
            preload = " ".join(filter(None, [str(library), os.environ.get("LD_PRELOAD")]))
            return {"LD_PRELOAD": preload}
    return {}


def ncu(archive, ncu_args, application, overrides):
    executable = _resolve_executable("ncu")
    application = _worker_python(application)
    overrides = {**_pinned_nvrtc(overrides), **overrides}
    with tempfile.TemporaryDirectory(prefix="kcoral_ncu_") as root:
        workdir = Path(root) / "inputs"
        workdir.mkdir()
        unpack_inputs(archive, workdir)
        report = Path(root) / "capture.ncu-rep"
        completed = subprocess.run(
            [executable, "--config-file", "0", "--export", str(report), *ncu_args, *application],
            cwd=workdir,
            env=environment(workdir, overrides),
            stdin=subprocess.DEVNULL,
        )
        return {
            "returncode": completed.returncode,
            "report": report.read_bytes() if report.is_file() else None,
        }
