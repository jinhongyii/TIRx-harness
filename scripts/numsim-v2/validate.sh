#!/usr/bin/env bash
# Validate lowered Module JSON files with the Rust contract:
#   numsim_core::program::Module::from_json + Program::validate + postcard round trip.
# Usage: scripts/numsim-v2/validate.sh DIR_OR_FILES...
#
# The validator is built against a contract-only shim of numsim-core
# (program.rs, site.rs, numsim-types and the two enums they import), generated
# from the working tree, or from `git archive HEAD` with VALIDATE_FROM_HEAD=1.
# Engine modules edited concurrently by other workers are not compiled.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
work=${VALIDATE_WORK:-${TMPDIR:-/tmp}/numsim-v2-validate}
core=tirx_harness/src/tirx_harness/numsim/core-rs
snap="$work/src/core-rs"
rm -rf "$work/src" "$work/shim" "$work/tool" && mkdir -p "$work/src"
if [ "${VALIDATE_FROM_HEAD:-0}" = 1 ]; then
  git -C "$repo" archive HEAD "$core" | tar -x -C "$work/src"
  mv "$work/src/$core" "$snap"
else
  rsync -a --exclude target "$repo/$core/" "$snap/"
fi
python3 "$here/make_contract_shim.py" "$snap" "$work/shim"
cp -r "$repo/$core/tools/validate-program" "$work/tool"
sed -i "s#numsim-core = { path = \"../../numsim-core\" }#numsim-core = { path = \"$work/shim\" }#" "$work/tool/Cargo.toml"
(cd "$work/tool" && CARGO_TARGET_DIR="$work/target" cargo build --release -q)
files=()
for arg in "$@"; do
  if [ -d "$arg" ]; then files+=("$arg"/*.json); else files+=("$arg"); fi
done
printf '%s\n' "${files[@]}" | "$work/target/release/validate-program" -
