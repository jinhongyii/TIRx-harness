#!/usr/bin/env bash
# Validate lowered Module JSON files with the Rust contract:
#   numsim_core::program::Module::from_json + Program::validate + postcard round trip.
# Usage: scripts/numsim-v2/validate.sh DIR_OR_FILES...
#
# Builds core-rs/tools/validate-program against a snapshot of core-rs
# (working tree by default, `git archive HEAD` with VALIDATE_FROM_HEAD=1).
# Other workers edit the analysis modules concurrently; if the snapshot does
# not compile, the modules the program decoder does not need (racecheck,
# synccheck) are replaced by empty stubs and the build is retried.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
work=${VALIDATE_WORK:-${TMPDIR:-/tmp}/numsim-v2-validate}
core=tirx_harness/src/tirx_harness/numsim/core-rs
snap="$work/src/core-rs"
rm -rf "$work/src" && mkdir -p "$work/src"
if [ "${VALIDATE_FROM_HEAD:-0}" = 1 ]; then
  git -C "$repo" archive HEAD "$core" | tar -x -C "$work/src"
  mv "$work/src/$core" "$snap"
  rm -rf "$snap/tools/validate-program" && mkdir -p "$snap/tools"
  cp -r "$repo/$core/tools/validate-program" "$snap/tools/"
else
  rsync -a --exclude target "$repo/$core/" "$snap/"
fi
build() { (cd "$snap/tools/validate-program" && CARGO_TARGET_DIR="$work/target" cargo build --release -q); }
if ! build 2>"$work/build.log"; then
  for module in racecheck synccheck; do
    src="$snap/numsim-core/src"
    rm -rf "$src/$module" "$src/$module.rs"
    echo "//! stubbed: not needed to decode and validate a Program" > "$src/$module.rs"
  done
  build
fi
files=()
for arg in "$@"; do
  if [ -d "$arg" ]; then files+=("$arg"/*.json); else files+=("$arg"); fi
done
printf '%s\n' "${files[@]}" | "$work/target/release/validate-program" -
