# Installation

```{container} lead
Install the components for your own agent workflow or a kcoral GPU server.
```

For [optimization runs](optimization-runs.md), setup prepares the packages and
skills automatically.

To run an exported kernel with `tvm-ffi`, follow
[Export a kernel](development/export-kernel.md) for the consumer dependencies.

## Before you start

- Linux x86_64, Python 3.12 or 3.13, and pip 25.1 or later.
- The CUDA Toolkit (`nvcc`, `ptxas`) where GPU kernels are compiled, and a
  compatible NVIDIA driver where they are run.
- To build from source: Cargo, **Rust 1.89.0 or later**, and a C linker. The
  NumSim engine (`numsim_core_py`) is compiled once, when the package is
  built; it does not compile each kernel, so running NumSim, Synccheck, or
  Racecheck needs no Rust toolchain.

## Install Python packages

Activate your agent's Python environment and choose one method.

### Install from PyPI

Install the released package without cloning this repository:

```bash
python -m pip install tirx-harness
```

Wheels include the compiled NumSim engine. If pip builds from a source
distribution, the build tools below are required.

### Build from source

Source builds require Git, a C linker, Python development headers, and the
Rust toolchain listed above.

```bash
git clone https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
python -m pip install .
```

`pip install .` compiles the NumSim engine, the Rust workspace
`tirx_harness/src/tirx_harness/numsim/core-rs`, into the Python extension
`tirx_harness.numsim.v2.numsim_core_py` (`cargo build --release --locked`; the
first build takes a few minutes).

#### Rebuild the engine in an editable checkout

An editable checkout, such as the uv one below, does not run that build step.
Build the extension into the source tree, and rerun the script after changing
anything under `core-rs`:

```bash
bash tirx_harness/src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh
```

Pass `--debug` for a debug build, and set `PY` to choose another Python
interpreter. To keep a private build next to a shared checkout, pass
`--out <dir>` with `CARGO_TARGET_DIR=<dir>/target`
([development loop](development/dev-loop.md)).

#### Optional: install with uv

From the initialized checkout, build the harness with dependencies from `uv.lock`:

```bash
uv sync --locked
source .venv/bin/activate
```

### Verify the installation

After any of these methods, check imports and that the engine loads (this
does not run checks or GPU kernels):

```bash
python -c "import tvm.tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness; print('Core imports OK')"
python -c "from tirx_harness.numsim.v2.compile import native; native(); print('Engine OK')"
```

## Install kcoral server dependencies

From the repository root, run `python -m pip install --group server`, or
`uv sync --locked --only-group server` and activate `.venv`. These install only
the server dependencies. See {ref}`remote execution <select-remote-execution>`
to start the server.

## Install agent skills

| Skill | Purpose |
| --- | --- |
| {repo}`tirx-wiki <skills/tirx-wiki/SKILL.md>` | Find TIRx-lite APIs, canonical kernels, and GPU references. |
| {repo}`tirx-debug-kernel <skills/tirx-debug-kernel/SKILL.md>` | Check correctness, investigate findings, and verify fixes. |
| {repo}`tirx-profile-kernel <skills/tirx-profile-kernel/SKILL.md>` | Measure performance and use profiler evidence to guide changes. |

The package bundles the skills. Install them into the directory your agent
reads, such as `.agents/skills` or `.claude/skills`:

```bash
tirx-harness skills install --dest /absolute/path/to/your/project/.agents/skills
```

The command also downloads the wiki manuals and reference repositories,
including `tirx-kernels`. Use `--no-fetch` to skip the download and `--force`
to replace an earlier installation. Editable installs, such as `uv sync`, do
not bundle the skills; copy them from `skills/` in the checkout instead.

Continue to [Quick Start](quick-start.md) for a concrete example.
