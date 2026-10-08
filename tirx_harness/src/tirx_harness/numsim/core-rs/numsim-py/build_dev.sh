#!/usr/bin/env bash
# Dev install of the `numsim_core_py` extension into the source tree, where
# `tirx_harness.numsim.v2` (an editable install) imports it from.
#
#   source scripts/dev-env.sh
#   bash tirx_harness/src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh [--debug]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
core="$(dirname "$here")"
dest="$(dirname "$core")/v2/numsim_core_py.abi3.so"
profile=release
flags=(--release)
if [[ "${1:-}" == "--debug" ]]; then profile=debug; flags=(); fi
python="${PY:-python3}"
(cd "$core" && PYO3_PYTHON="$python" cargo build "${flags[@]}" -p numsim-py --features extension-module)
cp "$core/target/$profile/libnumsim_core_py.so" "$dest.tmp.$$"
mv "$dest.tmp.$$" "$dest"
echo "installed $dest"
