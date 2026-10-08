#!/usr/bin/env bash
# Dev install of the `numsim_core_py` extension.
#
#   source scripts/dev-env.sh
#   bash tirx_harness/src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh [--debug] [--out DIR]
#
# Without --out the extension is installed into the source tree, where
# `tirx_harness.numsim.v2` (an editable install) imports it from. Concurrent
# builds must not share a cargo target dir or the shared .so: use
# CARGO_TARGET_DIR=<private dir> and --out <private dir>, then run pytest with
#   -o "pythonpath=<private dir>/pkg ."   (see docs/development/dev-loop.md)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
core="$(dirname "$here")"
v2dir="$(dirname "$core")/v2"
profile=release
flags=(--release)
out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) profile=debug; flags=(); shift ;;
    --out) out="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
python="${PY:-python3}"
target="${CARGO_TARGET_DIR:-$core/target}"
(cd "$core" && CARGO_TARGET_DIR="$target" PYO3_PYTHON="$python" cargo build "${flags[@]}" -p numsim-py --features extension-module)
if [[ -n "$out" ]]; then
  # Private install: a copy of the v2 package next to a private .so, importable
  # ahead of the source tree via pytest -o "pythonpath=$out/pkg .".
  pkg="$out/pkg/tirx_harness/numsim/v2"
  mkdir -p "$pkg"
  cp -r "$v2dir"/*.py "$v2dir"/lowering "$pkg"/
  dest="$pkg/numsim_core_py.abi3.so"
else
  dest="$v2dir/numsim_core_py.abi3.so"
fi
cp "$target/$profile/libnumsim_core_py.so" "$dest.tmp.$$"
mv "$dest.tmp.$$" "$dest"
echo "installed $dest"
