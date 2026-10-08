"""Numerical correctness for every single-GPU canonical registry kernel."""

from __future__ import annotations

from time import perf_counter

import pytest

from tirx_kernels.registry import discover_kernels
from tirx_harness.numsim import Engine, run_case, transpile
from tests.numsim.corpus.canonical_cases import (
    CANONICAL_KERNEL_CASES,
    CANONICAL_KERNEL_MANIFEST,
    MULTI_GPU_ONLY_CANONICAL_KERNELS,
)
from tests.numsim.corpus.kernels.deepgemm import prepare_mega_moe_case
from tests.numsim.support._tirx_kernels import load_tirx_kernel


_MEGA_MOE_MAX_CONFIG_LABEL = "t8192_m8192_h7168_i3072_e384_k6_g1"
# 2026-09-18: this exact Engine.run took 130.602536/136.797175s on the 224-CPU runner,
# with 16 engine workers and pytest -n16/worksteal alongside the Racecheck gate.
# Both runs passed idle-host preflight. Keep the existing 140s guard despite
# its narrow 2.3% headroom; do not widen it to fit a nominal noise margin.
# Preparation/build are excluded; kernels use d11cb9e.
_MEGA_MOE_NUMSIM_WALL_TIME_LIMIT_SECONDS = 140.0


def test_canonical_manifest_exactly_matches_single_gpu_registry() -> None:
    discovered = set(discover_kernels(strict=True))
    manifested = {case.name for case in CANONICAL_KERNEL_MANIFEST}

    assert len(CANONICAL_KERNEL_MANIFEST) == len(manifested)
    assert manifested.isdisjoint(MULTI_GPU_ONLY_CANONICAL_KERNELS)
    assert discovered == manifested | MULTI_GPU_ONLY_CANONICAL_KERNELS


