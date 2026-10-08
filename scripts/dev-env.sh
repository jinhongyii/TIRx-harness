# Source from any directory: `source scripts/dev-env.sh`.
# Install with: uv sync --locked --extra test --group benchmark --inexact  (corpus kernels import torch)
#
# Every path is an overridable default: export the variable (or, for the
# library path, NUMSIM_LD_LIBRARY_PATH) before sourcing to change it.

_numsim_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"

# The login shell points these at a local TVM 0.26 development tree, which
# shadows the TIRx-enabled `tvm` installed in the venv, so `import tvm` loads
# the wrong build.
unset PYTHONPATH TVM_LIBRARY_PATH TVM_HOME

export PY="${PY:-$_numsim_repo/.venv/bin/python}"
export NUMSIM_CACHE_DIR="${NUMSIM_CACHE_DIR:-$HOME/.cache/numsim-refactor}"

# Runtime libraries come from the venv: the NVIDIA wheels (nvidia/*/lib), torch
# and TVM. LD_LIBRARY_PATH itself is replaced, not extended, because the login
# shell's value points at the TVM development tree; set NUMSIM_LD_LIBRARY_PATH
# to use your own value instead.
_numsim_site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
_numsim_ld=""
for _numsim_dir in "$_numsim_site"/nvidia/*/lib "$_numsim_site/torch/lib" "$_numsim_site/tvm/lib" "$_numsim_site/tvm/lib64"; do
    [ -d "$_numsim_dir" ] && _numsim_ld="${_numsim_ld:+$_numsim_ld:}$_numsim_dir"
done
export LD_LIBRARY_PATH="${NUMSIM_LD_LIBRARY_PATH:-$_numsim_ld}"
unset _numsim_repo _numsim_site _numsim_ld _numsim_dir
